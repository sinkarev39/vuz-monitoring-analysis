# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль softmax_numpy — многоклассовая логистическая регрессия «с нуля».

Модель реализована только средствами NumPy, без готовых алгоритмов обучения:
прямой проход, функция потерь, аналитический градиент и шаг градиентного
спуска записаны явно. Она служит опорной (базовой) моделью, с которой
сравнивается нейронная сеть, и хранится в файле model3.npz.

Математическая постановка:
    z = XW + b,   p_k = exp(z_k) / sum_j exp(z_j)            (softmax)
    L = -1/N * sum_i w_{y_i} * log p_{i, y_i} + lambda/2 * ||W||^2
    dL/dW = X^T (P - Y) / N + lambda * W,   dL/db = mean(P - Y)
"""

import numpy as np


def softmax(logits: np.ndarray) -> np.ndarray:
    """Устойчивое вычисление softmax: из каждой строки вычитается её максимум."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exponent = np.exp(shifted)
    return exponent / exponent.sum(axis=1, keepdims=True)


class SoftmaxRegressionGD:
    """Многоклассовая логистическая регрессия, обучаемая мини-пакетным градиентным спуском."""

    def __init__(self, n_classes: int = 3, learning_rate: float = 0.1, epochs: int = 200,
                 batch_size: int = 128, l2: float = 1e-3, momentum: float = 0.9,
                 class_weighted: bool = True, seed: int = 42):
        self.n_classes = n_classes
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.l2 = l2                       # коэффициент L2-регуляризации (lambda)
        self.momentum = momentum           # коэффициент инерции (метод моментов)
        self.class_weighted = class_weighted
        self.seed = seed
        self.weights = None                # матрица W: признаки x классы
        self.bias = None                   # вектор смещений b
        self.history = []                  # значение функции потерь по эпохам

    def _loss(self, features, onehot, sample_weights) -> float:
        probabilities = softmax(features @ self.weights + self.bias)
        cross_entropy = -(sample_weights * np.log(np.clip((probabilities * onehot).sum(axis=1), 1e-12, 1))).mean()
        return float(cross_entropy + 0.5 * self.l2 * (self.weights ** 2).sum())

    def fit(self, features: np.ndarray, labels: np.ndarray, val_features=None, val_labels=None):
        """Обучение модели. При наличии валидационной выборки запоминается и её ошибка."""
        rng = np.random.default_rng(self.seed)
        features = np.asarray(features, dtype=np.float64)
        labels = np.asarray(labels, dtype=int)
        n_samples, n_features = features.shape
        onehot = np.eye(self.n_classes)[labels]

        # Веса объектов: редким классам придаётся больший вес
        if self.class_weighted:
            counts = np.bincount(labels, minlength=self.n_classes).astype(float)
            class_weights = n_samples / (self.n_classes * np.maximum(counts, 1.0))
        else:
            class_weights = np.ones(self.n_classes)
        sample_weights = class_weights[labels]

        self.weights = rng.normal(0.0, 0.01, size=(n_features, self.n_classes))
        self.bias = np.zeros(self.n_classes)
        velocity_w, velocity_b = np.zeros_like(self.weights), np.zeros_like(self.bias)
        self.history = []

        for epoch in range(1, self.epochs + 1):
            order = rng.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                batch = order[start:start + self.batch_size]
                x, y, w = features[batch], onehot[batch], sample_weights[batch][:, None]
                probabilities = softmax(x @ self.weights + self.bias)
                # Градиент перекрёстной энтропии по логитам равен (P - Y)
                delta = (probabilities - y) * w / len(batch)
                grad_w = x.T @ delta + self.l2 * self.weights
                grad_b = delta.sum(axis=0)
                velocity_w = self.momentum * velocity_w - self.learning_rate * grad_w
                velocity_b = self.momentum * velocity_b - self.learning_rate * grad_b
                self.weights += velocity_w
                self.bias += velocity_b
            record = {"epoch": epoch, "train_loss": self._loss(features, onehot, sample_weights)}
            if val_features is not None:
                val_onehot = np.eye(self.n_classes)[np.asarray(val_labels, dtype=int)]
                record["val_loss"] = self._loss(np.asarray(val_features, dtype=np.float64), val_onehot,
                                                class_weights[np.asarray(val_labels, dtype=int)])
            self.history.append(record)
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Вероятности классов для новых объектов."""
        return softmax(np.asarray(features, dtype=np.float64) @ self.weights + self.bias)

    def predict(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba(features).argmax(axis=1)

    # ------------------------------------------------------- сохранение
    def save(self, path) -> None:
        """Сохраняет параметры модели в файл формата .npz."""
        np.savez(path, weights=self.weights, bias=self.bias,
                 params=np.array([self.n_classes, self.learning_rate, self.epochs, self.batch_size,
                                  self.l2, self.momentum, int(self.class_weighted), self.seed], dtype=float))

    @classmethod
    def load(cls, path) -> "SoftmaxRegressionGD":
        """Загружает модель из файла .npz."""
        data = np.load(path)
        params = data["params"]
        model = cls(int(params[0]), float(params[1]), int(params[2]), int(params[3]), float(params[4]),
                    float(params[5]), bool(params[6]), int(params[7]))
        model.weights, model.bias = data["weights"], data["bias"]
        return model
