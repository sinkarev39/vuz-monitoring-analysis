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


class MonitoringMLP(nn.Module):
    """
    Многослойный перцептрон для классификации табличных данных.

    Вход:  вектор признаков организации за год t (после предобработки).
    Выход: логиты трёх классов результативности в году t+1; вероятности
           получаются применением softmax.
    """

    def __init__(self, n_features: int, n_classes: int = 3, hidden=(128, 64), dropout: float = 0.3,
                 batch_norm: bool = True, activation: str = "relu"):
        super().__init__()
        self.config = {"n_features": n_features, "n_classes": n_classes, "hidden": list(hidden),
                       "dropout": dropout, "batch_norm": batch_norm, "activation": activation}
        layers, width = [], n_features
        for size in hidden:
            layers += _block(width, size, activation, batch_norm, dropout)
            width = size
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(width, n_classes)
        self._init_weights()

    def _init_weights(self) -> None:
        """Инициализация Хе (Kaiming): подходит для слоёв с ReLU-подобной активацией."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.body(x))

    def count_parameters(self) -> int:
        """Число обучаемых параметров сети."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


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
