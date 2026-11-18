# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль networks — архитектуры нейронных сетей системы (PyTorch).

  MonitoringMLP          — многослойный перцептрон для прогноза класса
                           результативности вуза (файл модели model1.pt);
  MonitoringAutoencoder  — автокодировщик для сжатого представления профиля
                           вуза, поиска нетипичных записей и построения
                           типологических групп (файл модели model2.pt).
"""

import torch
from torch import nn

# Доступные функции активации скрытых слоёв
ACTIVATIONS = {"relu": nn.ReLU, "gelu": nn.GELU, "elu": nn.ELU, "tanh": nn.Tanh}


def _block(in_features: int, out_features: int, activation: str, batch_norm: bool, dropout: float) -> list:
    """Один скрытый блок: линейный слой -> нормализация -> активация -> прореживание."""
    layers = [nn.Linear(in_features, out_features)]
    if batch_norm:
        layers.append(nn.BatchNorm1d(out_features))
    layers.append(ACTIVATIONS[activation]())
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    return layers


# Границы классов по числу выполненных показателей: 0–1, 2–3, 4–5
COUNT_TO_CLASS = (0, 0, 1, 1, 2, 2)


def count_distribution(pass_probabilities: torch.Tensor) -> torch.Tensor:
    """
    Распределение числа выполненных показателей по вероятностям выполнения
    каждого из них (распределение Пуассона — биномиальное).

    Вход:  тензор N x K — вероятности выполнения K показателей;
    выход: тензор N x (K+1) — вероятности того, что выполнено 0, 1, ..., K.
    Вычисляется последовательной свёрткой: добавляя очередной показатель,
    «сдвигаем» распределение на единицу с вероятностью p и оставляем с 1 - p.
    """
    n_rows, n_items = pass_probabilities.shape
    distribution = torch.zeros(n_rows, n_items + 1, dtype=pass_probabilities.dtype)
    distribution[:, 0] = 1.0
    for item in range(n_items):
        p = pass_probabilities[:, item:item + 1]
        shifted = torch.cat([torch.zeros(n_rows, 1, dtype=distribution.dtype), distribution[:, :-1]], dim=1)
        distribution = distribution * (1.0 - p) + shifted * p
    return distribution


class MonitoringMLP(nn.Module):
    """
    Многослойный перцептрон для прогноза класса результативности.

    Вход:  вектор признаков организации за год t (после предобработки).
    Выходы:
      * основной — логиты трёх классов результативности в году t+1;
      * вспомогательный (если n_aux > 0) — логиты выполнения каждого из
        основных показателей в году t+1 (многозадачное обучение).
    Итоговые вероятности классов — взвешенная сумма двух оценок: softmax
    основного выхода и распределения числа выполненных показателей,
    пересчитанного в классы («структурная» оценка, опирающаяся на устройство
    самой задачи). Вес структурной оценки задаётся параметром structure_weight.
    """

    def __init__(self, n_features: int, n_classes: int = 3, hidden=(128, 64), dropout: float = 0.3,
                 batch_norm: bool = True, activation: str = "relu", n_aux: int = 0,
                 structure_weight: float = 0.75):
        super().__init__()
        self.config = {"n_features": n_features, "n_classes": n_classes, "hidden": list(hidden),
                       "dropout": dropout, "batch_norm": batch_norm, "activation": activation,
                       "n_aux": n_aux, "structure_weight": structure_weight}
        self.structure_weight = structure_weight
        layers, width = [], n_features
        for size in hidden:
            layers += _block(width, size, activation, batch_norm, dropout)
            width = size
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_classes)
        self.aux_head = nn.Linear(width, n_aux) if n_aux > 0 else None
        self._init_weights()

    def _init_weights(self) -> None:
        """Инициализация Хе (Kaiming): подходит для слоёв с ReLU-подобной активацией."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Логиты классов (основной выход)."""
        return self.head(self.body(x))

    def forward_all(self, x: torch.Tensor):
        """Оба выхода сети: (логиты классов, логиты выполнения показателей или None)."""
        hidden = self.body(x)
        return self.head(hidden), (self.aux_head(hidden) if self.aux_head is not None else None)

    def class_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Итоговые вероятности классов с учётом вспомогательного выхода."""
        logits, aux_logits = self.forward_all(x)
        probabilities = torch.softmax(logits, dim=1)
        if aux_logits is None or aux_logits.shape[1] + 1 != len(COUNT_TO_CLASS):
            return probabilities
        counts = count_distribution(torch.sigmoid(aux_logits))
        structural = torch.zeros_like(probabilities)
        for count, label in enumerate(COUNT_TO_CLASS):
            structural[:, label] += counts[:, count]
        return self.structure_weight * structural + (1.0 - self.structure_weight) * probabilities

    def count_parameters(self) -> int:
        """Число обучаемых параметров сети."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class MonitoringEnsemble(nn.Module):
    """
    Ансамбль одинаковых по архитектуре сетей MonitoringMLP, обученных с
    разными начальными весами и порядком объектов. Вероятности классов
    усредняются по участникам: случайные ошибки отдельных сетей взаимно
    гасятся, и прогноз становится устойчивее.
    """

    def __init__(self, members: list):
        super().__init__()
        self.members = nn.ModuleList(members)
        first = members[0].config
        self.config = {"ensemble": [dict(member.config) for member in members],
                       "n_classes": first["n_classes"], "n_features": first["n_features"],
                       "hidden": first["hidden"], "n_aux": first.get("n_aux", 0)}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Усреднённые логиты классов."""
        return torch.stack([member(x) for member in self.members]).mean(dim=0)

    def class_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        """Усреднённые по участникам вероятности классов."""
        return torch.stack([member.class_probabilities(x) for member in self.members]).mean(dim=0)

    def count_parameters(self) -> int:
        return sum(member.count_parameters() for member in self.members)


def build_classifier(config: dict) -> nn.Module:
    """Создаёт сеть (или ансамбль сетей) по сохранённому описанию архитектуры."""
    if "ensemble" in config:
        return MonitoringEnsemble([build_classifier(member) for member in config["ensemble"]])
    return MonitoringMLP(config["n_features"], config["n_classes"], tuple(config["hidden"]),
                         config["dropout"], config["batch_norm"], config["activation"],
                         int(config.get("n_aux", 0)), float(config.get("structure_weight", 0.5)))


class MonitoringAutoencoder(nn.Module):
    """
    Симметричный автокодировщик.

    Кодировщик сжимает вектор признаков до скрытого представления малой
    размерности, декодировщик восстанавливает исходный вектор. Большая
    ошибка восстановления означает, что профиль организации не похож на
    типичные, то есть запись нетипична (или в отчётных данных ошибка).
    """

    def __init__(self, n_features: int, hidden=(64, 24), latent_dim: int = 8, dropout: float = 0.0,
                 activation: str = "relu"):
        super().__init__()
        self.config = {"n_features": n_features, "hidden": list(hidden), "latent_dim": latent_dim,
                       "dropout": dropout, "activation": activation}
        encoder, width = [], n_features
        for size in hidden:
            encoder += _block(width, size, activation, True, dropout)
            width = size
        encoder.append(nn.Linear(width, latent_dim))
        self.encoder = nn.Sequential(*encoder)

        decoder, width = [], latent_dim
        for size in reversed(hidden):
            decoder += _block(width, size, activation, True, dropout)
            width = size
        decoder.append(nn.Linear(width, n_features))
        self.decoder = nn.Sequential(*decoder)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Скрытое (сжатое) представление объекта."""
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
