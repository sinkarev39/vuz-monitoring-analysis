# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль registry — сохранение обученных моделей в файлы и их загрузка.

Файл модели (*.pt) — это словарь, в котором лежат:
  * веса нейронной сети (state_dict) и параметры её архитектуры;
  * состояние предобработчика признаков;
  * гиперпараметры обучения, метрики качества и история по эпохам;
  * сведения об авторе и теме работы.
В файле нет исполняемого кода, только тензоры и простые типы данных,
поэтому он загружается в безопасном режиме (weights_only=True).
"""

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import torch

from vuzmon import __author__
from vuzmon.ml.networks import MonitoringAutoencoder, MonitoringMLP
from vuzmon.ml.preprocessing import TabularPreprocessor

TOPIC = "Разработка системы интеллектуального анализа данных мониторинга вузов"
FORMAT_CLASSIFIER = "vuzmon-mlp-v1"
FORMAT_AUTOENCODER = "vuzmon-autoencoder-v1"


@dataclass
class LoadedModel:
    """Загруженная модель вместе с предобработчиком и сопроводительными сведениями."""
    model: torch.nn.Module
    preprocessor: TabularPreprocessor
    meta: dict
    path: Path = None

    @property
    def kind(self) -> str:
        return "mlp" if self.meta.get("format") == FORMAT_CLASSIFIER else "autoencoder"


def _base_payload(model, preprocessor: TabularPreprocessor, file_format: str, extra: dict) -> dict:
    payload = {
        "format": file_format,
        "author": __author__,
        "topic": TOPIC,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "torch_version": torch.__version__,
        "model_config": dict(model.config),
        "state_dict": model.state_dict(),
        "preprocessor": preprocessor.state_dict(),
    }
    payload.update(extra or {})
    return payload


def save_classifier(path, model: MonitoringMLP, preprocessor: TabularPreprocessor, train_config: dict,
                    metrics: dict, class_names, history: list = None, extra: dict = None) -> Path:
    """Сохраняет классификатор со всем, что нужно для его применения и проверки."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _base_payload(model, preprocessor, FORMAT_CLASSIFIER, {
        "train_config": dict(train_config), "metrics": metrics, "class_names": list(class_names),
        "history": list(history or []), **(extra or {})})
    torch.save(payload, path)
    return path


def save_autoencoder(path, model: MonitoringAutoencoder, preprocessor: TabularPreprocessor,
                     metrics: dict = None, history: list = None, extra: dict = None) -> Path:
    """Сохраняет автокодировщик (в extra обычно кладутся центры кластеров и порог нетипичности)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _base_payload(model, preprocessor, FORMAT_AUTOENCODER, {
        "metrics": metrics or {}, "history": list(history or []), **(extra or {})})
    torch.save(payload, path)
    return path


def load_model(path) -> LoadedModel:
    """
    Загружает модель из файла и восстанавливает сеть нужной архитектуры.
    Тип модели определяется по полю 'format' внутри файла.
    """
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "format" not in payload:
        raise ValueError("Файл не является моделью системы анализа данных мониторинга")
    config = payload["model_config"]
    if payload["format"] == FORMAT_CLASSIFIER:
        model = MonitoringMLP(config["n_features"], config["n_classes"], tuple(config["hidden"]),
                              config["dropout"], config["batch_norm"], config["activation"])
    elif payload["format"] == FORMAT_AUTOENCODER:
        model = MonitoringAutoencoder(config["n_features"], tuple(config["hidden"]), config["latent_dim"],
                                      config["dropout"], config["activation"])
    else:
        raise ValueError(f"Неизвестный формат файла модели: {payload['format']}")
    model.load_state_dict(payload["state_dict"])
    model.eval()
    preprocessor = TabularPreprocessor.from_state(payload["preprocessor"])
    meta = {key: value for key, value in payload.items() if key not in ("state_dict", "preprocessor")}
    return LoadedModel(model=model, preprocessor=preprocessor, meta=meta, path=path)


def next_model_path(models_dir, extension: str = ".pt") -> Path:
    """Свободное имя файла вида model<N>.pt в каталоге моделей."""
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    number = 1
    while any(models_dir.glob(f"model{number}.*")):
        number += 1
    return models_dir / f"model{number}{extension}"
