# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль trainer — обучение нейронных сетей (PyTorch).

Цикл обучения написан вручную: проход по мини-пакетам, вычисление функции
потерь, обратное распространение ошибки, шаг оптимизатора, оценка на
валидационной выборке после каждой эпохи, изменение шага обучения и ранняя
остановка. После каждой эпохи вызывается функция обратного вызова — через
неё веб-интерфейс показывает ход обучения и сохраняет метрики в базу.
"""

import copy
from dataclasses import asdict, dataclass, field

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from vuzmon.ml import metrics
from vuzmon.ml.networks import MonitoringAutoencoder, MonitoringEnsemble, MonitoringMLP


@dataclass
class TrainConfig:
    """Гиперпараметры обучения классификатора."""
    hidden: tuple = (128, 64)        # размеры скрытых слоёв
    dropout: float = 0.3             # вероятность прореживания
    batch_norm: bool = True          # пакетная нормализация
    activation: str = "relu"
    optimizer: str = "adamw"         # sgd / adam / adamw
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4       # L2-регуляризация весов
    batch_size: int = 64
    epochs: int = 100
    patience: int = 15               # эпох без улучшения до ранней остановки (0 — выключено)
    label_smoothing: float = 0.0     # сглаживание меток
    class_weighted: bool = False     # веса классов в функции потерь (повышает полноту редкого класса)
    aux_weight: float = 2.0          # вес вспомогательной задачи; 0 — обычная однозадачная сеть
    structure_weight: float = 0.75   # доля структурной оценки в итоговых вероятностях классов
    ensemble_size: int = 1           # число сетей в ансамбле (1 — одна сеть)
    scheduler: str = "plateau"       # none / plateau / cosine
    seed: int = 42

    def to_dict(self) -> dict:
        data = asdict(self)
        data["hidden"] = list(self.hidden)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "TrainConfig":
        known = {key: value for key, value in data.items() if key in cls.__dataclass_fields__}
        if "hidden" in known:
            known["hidden"] = tuple(int(v) for v in known["hidden"])
        return cls(**known)


@dataclass
class TrainResult:
    """Результат обучения: модель, история по эпохам и номер лучшей эпохи."""
    model: nn.Module
    history: list = field(default_factory=list)
    best_epoch: int = 0
    stopped_early: bool = False


def set_seed(seed: int) -> None:
    """Фиксирует генераторы случайных чисел — обучение становится воспроизводимым."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def _make_optimizer(model: nn.Module, config: TrainConfig) -> torch.optim.Optimizer:
    if config.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.learning_rate, momentum=0.9,
                               weight_decay=config.weight_decay)
    if config.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=config.learning_rate,
                                weight_decay=config.weight_decay)
    return torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)


def _make_scheduler(optimizer, config: TrainConfig):
    if config.scheduler == "plateau":
        # Шаг обучения уменьшается вдвое, если ошибка на валидации перестала падать
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
    if config.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.epochs)
    return None


def _loader(features: np.ndarray, targets: np.ndarray, batch_size: int, shuffle: bool, seed: int, extra=None):
    """Загрузчик мини-пакетов поверх массивов NumPy (extra — дополнительные цели)."""
    tensors = [torch.as_tensor(features, dtype=torch.float32), torch.as_tensor(targets)]
    if extra is not None:
        tensors.append(torch.as_tensor(extra, dtype=torch.float32))
    dataset = TensorDataset(*tensors)
    generator = torch.Generator().manual_seed(seed)
    # Неполный последний пакет из одного объекта ломает пакетную нормализацию — отбрасываем его
    drop_last = shuffle and len(dataset) % batch_size == 1
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, generator=generator, drop_last=drop_last)


@torch.no_grad()
def predict_proba(model: nn.Module, features: np.ndarray, batch_size: int = 1024) -> np.ndarray:
    """Вероятности классов для матрицы признаков."""
    model.eval()
    parts = []
    tensor = torch.as_tensor(features, dtype=torch.float32)
    for start in range(0, len(tensor), batch_size):
        parts.append(model.class_probabilities(tensor[start:start + batch_size]).numpy())
    return np.concatenate(parts) if parts else np.empty((0, model.config["n_classes"]))


def train_classifier(x_train, y_train, x_val, y_val, config: TrainConfig, on_epoch=None,
                     aux_train=None, aux_val=None) -> TrainResult:
    """
    Обучает классификатор: одну сеть или ансамбль из config.ensemble_size сетей.

    Участники ансамбля отличаются только зерном генератора случайных чисел
    (начальные веса и порядок объектов). В результат попадает история
    обучения первой сети; функция on_epoch вызывается тоже только для неё.
    """
    if config.ensemble_size <= 1:
        return _train_single(x_train, y_train, x_val, y_val, config, on_epoch, aux_train, aux_val)
    members, first = [], None
    for index in range(config.ensemble_size):
        member_config = TrainConfig.from_dict({**config.to_dict(), "seed": config.seed + index,
                                               "ensemble_size": 1})
        outcome = _train_single(x_train, y_train, x_val, y_val, member_config,
                                on_epoch if index == 0 else None, aux_train, aux_val)
        members.append(outcome.model)
        first = first or outcome
    ensemble = MonitoringEnsemble(members)
    ensemble.eval()
    return TrainResult(model=ensemble, history=first.history, best_epoch=first.best_epoch,
                       stopped_early=first.stopped_early)


def _train_single(x_train, y_train, x_val, y_val, config: TrainConfig, on_epoch=None,
                  aux_train=None, aux_val=None) -> TrainResult:
    """
    Обучает одну сеть MonitoringMLP.

    aux_train, aux_val — вспомогательные цели (выполнение каждого показателя
    в следующем году). Если они переданы и config.aux_weight > 0, сеть
    обучается как многозадачная: к перекрёстной энтропии по классам
    добавляется бинарная перекрёстная энтропия по показателям с весом aux_weight.

    on_epoch(record) — необязательная функция, вызываемая после каждой эпохи
    со словарём метрик; если она возвращает False, обучение прерывается.
    """
    set_seed(config.seed)
    y_train = np.asarray(y_train, dtype=np.int64)
    y_val = np.asarray(y_val, dtype=np.int64)
    n_classes = int(max(y_train.max(), y_val.max())) + 1
    multitask = aux_train is not None and aux_val is not None and config.aux_weight > 0
    n_aux = int(np.asarray(aux_train).shape[1]) if multitask else 0
    model = MonitoringMLP(x_train.shape[1], n_classes, config.hidden, config.dropout, config.batch_norm,
                          config.activation, n_aux, config.structure_weight)

    # Веса классов обратно пропорциональны их частоте в обучающей выборке
    weight = None
    if config.class_weighted:
        counts = np.bincount(y_train, minlength=n_classes).astype(np.float64)
        weight = torch.as_tensor(len(y_train) / (n_classes * np.maximum(counts, 1.0)), dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=weight, label_smoothing=config.label_smoothing)
    eval_criterion = nn.CrossEntropyLoss(weight=weight)
    aux_criterion = nn.BCEWithLogitsLoss()
    optimizer = _make_optimizer(model, config)
    scheduler = _make_scheduler(optimizer, config)
    train_loader = _loader(x_train, y_train, config.batch_size, True, config.seed,
                           aux_train if multitask else None)
    x_val_tensor = torch.as_tensor(x_val, dtype=torch.float32)
    y_val_tensor = torch.as_tensor(y_val)
    aux_val_tensor = torch.as_tensor(aux_val, dtype=torch.float32) if multitask else None

    result = TrainResult(model=model)
    best_loss, best_state, epochs_without_gain = float("inf"), None, 0

    for epoch in range(1, config.epochs + 1):
        # ---------------------------------------------------- обучение
        model.train()
        total_loss, total_correct, total_seen = 0.0, 0, 0
        for batch in train_loader:
            batch_x, batch_y = batch[0], batch[1]
            optimizer.zero_grad()
            logits, aux_logits = model.forward_all(batch_x)
            loss = criterion(logits, batch_y)
            if multitask:
                loss = loss + config.aux_weight * aux_criterion(aux_logits, batch[2])
            loss.backward()                                   # обратное распространение ошибки
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)  # защита от «взрыва» градиентов
            optimizer.step()
            total_loss += loss.item() * len(batch_y)
            total_correct += int((logits.argmax(dim=1) == batch_y).sum())
            total_seen += len(batch_y)

        # ---------------------------------------------------- валидация
        model.eval()
        with torch.no_grad():
            val_logits, val_aux = model.forward_all(x_val_tensor)
            val_loss = float(eval_criterion(val_logits, y_val_tensor))
            if multitask:
                val_loss += config.aux_weight * float(aux_criterion(val_aux, aux_val_tensor))
            val_pred = model.class_probabilities(x_val_tensor).argmax(dim=1).numpy()
        val_scores = metrics.per_class_scores(metrics.confusion_matrix(y_val, val_pred, n_classes))
        record = {
            "epoch": epoch,
            "train_loss": total_loss / max(total_seen, 1),
            "val_loss": val_loss,
            "train_metric": total_correct / max(total_seen, 1),
            "val_metric": metrics.accuracy(y_val, val_pred),
            "val_f1": float(val_scores["f1"].mean()),
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        result.history.append(record)

        if scheduler is not None:
            if config.scheduler == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ------------------------------- запоминаем лучшее состояние сети
        if val_loss < best_loss - 1e-5:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
            result.best_epoch, epochs_without_gain = epoch, 0
        else:
            epochs_without_gain += 1

        if on_epoch is not None and on_epoch(record) is False:
            break
        if config.patience and epochs_without_gain >= config.patience:
            result.stopped_early = True
            break

    if best_state is not None:
        model.load_state_dict(best_state)   # возвращаем веса лучшей эпохи
    model.eval()
    return result


def train_autoencoder(x_train, x_val, hidden=(64, 24), latent_dim: int = 8, epochs: int = 150,
                      learning_rate: float = 1e-3, batch_size: int = 64, weight_decay: float = 1e-5,
                      noise_std: float = 0.1, patience: int = 20, seed: int = 42,
                      on_epoch=None) -> TrainResult:
    """
    Обучает шумоподавляющий автокодировщик: на вход подаётся объект с
    добавленным гауссовым шумом, восстанавливается исходный объект.
    Функция потерь — среднеквадратичная ошибка восстановления.
    """
    set_seed(seed)
    model = MonitoringAutoencoder(x_train.shape[1], hidden, latent_dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=6)
    criterion = nn.MSELoss()
    loader = _loader(x_train, np.zeros(len(x_train), dtype=np.int64), batch_size, True, seed)
    x_val_tensor = torch.as_tensor(x_val, dtype=torch.float32)

    result = TrainResult(model=model)
    best_loss, best_state, epochs_without_gain = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss, total_seen = 0.0, 0
        for batch_x, _ in loader:
            optimizer.zero_grad()
            noisy = batch_x + noise_std * torch.randn_like(batch_x)
            loss = criterion(model(noisy), batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)
            total_seen += len(batch_x)
        model.eval()
        with torch.no_grad():
            val_loss = float(criterion(model(x_val_tensor), x_val_tensor))
        scheduler.step(val_loss)
        record = {"epoch": epoch, "train_loss": total_loss / max(total_seen, 1), "val_loss": val_loss,
                  "train_metric": None, "val_metric": None,
                  "learning_rate": optimizer.param_groups[0]["lr"]}
        result.history.append(record)
        if val_loss < best_loss - 1e-6:
            best_loss, best_state = val_loss, copy.deepcopy(model.state_dict())
            result.best_epoch, epochs_without_gain = epoch, 0
        else:
            epochs_without_gain += 1
        if on_epoch is not None and on_epoch(record) is False:
            break
        if patience and epochs_without_gain >= patience:
            result.stopped_early = True
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    return result


@torch.no_grad()
def reconstruction_error(model: MonitoringAutoencoder, features: np.ndarray) -> np.ndarray:
    """Средний квадрат ошибки восстановления для каждого объекта (оценка нетипичности)."""
    model.eval()
    tensor = torch.as_tensor(features, dtype=torch.float32)
    return ((model(tensor) - tensor) ** 2).mean(dim=1).numpy()


@torch.no_grad()
def encode(model: MonitoringAutoencoder, features: np.ndarray) -> np.ndarray:
    """Скрытые представления объектов (выход кодировщика)."""
    model.eval()
    return model.encode(torch.as_tensor(features, dtype=torch.float32)).numpy()
