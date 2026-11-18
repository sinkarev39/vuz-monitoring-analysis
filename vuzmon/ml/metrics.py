# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль metrics — метрики качества классификации.

Основные метрики реализованы самостоятельно на NumPy (матрица ошибок,
точность, полнота, F-мера, сбалансированная доля верных ответов,
логарифмическая функция потерь, площадь под ROC-кривой). Это позволяет
использовать одни и те же формулы для всех моделей системы — и нейросетевых,
и классических — и не зависеть от особенностей сторонних реализаций.
"""

import numpy as np


def confusion_matrix(y_true, y_pred, n_classes: int) -> np.ndarray:
    """Матрица ошибок: строка — истинный класс, столбец — предсказанный."""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for true_label, predicted_label in zip(np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)):
        matrix[true_label, predicted_label] += 1
    return matrix


def accuracy(y_true, y_pred) -> float:
    """Доля верных ответов (accuracy)."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float((y_true == y_pred).mean()) if len(y_true) else 0.0


def per_class_scores(matrix: np.ndarray) -> dict:
    """Точность (precision), полнота (recall) и F1 по каждому классу."""
    true_positive = np.diag(matrix).astype(float)
    predicted = matrix.sum(axis=0).astype(float)   # сколько раз класс был предсказан
    actual = matrix.sum(axis=1).astype(float)      # сколько объектов класса на самом деле
    precision = np.divide(true_positive, predicted, out=np.zeros_like(true_positive), where=predicted > 0)
    recall = np.divide(true_positive, actual, out=np.zeros_like(true_positive), where=actual > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(true_positive),
                   where=denominator > 0)
    return {"precision": precision, "recall": recall, "f1": f1, "support": actual}


def log_loss(y_true, probabilities) -> float:
    """Логарифмическая функция потерь (перекрёстная энтропия) по вероятностям."""
    y_true = np.asarray(y_true, dtype=int)
    clipped = np.clip(np.asarray(probabilities, dtype=float), 1e-12, 1.0)
    return float(-np.log(clipped[np.arange(len(y_true)), y_true]).mean())


def binary_auc(labels, scores) -> float:
    """
    Площадь под ROC-кривой для бинарной задачи через ранги (статистика
    Манна — Уитни): вероятность того, что случайный положительный объект
    получит оценку выше случайного отрицательного.
    """
    labels, scores = np.asarray(labels, dtype=bool), np.asarray(scores, dtype=float)
    positives, negatives = int(labels.sum()), int((~labels).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Одинаковым оценкам назначается средний ранг
    sorted_scores = scores[order]
    start = 0
    while start < len(scores):
        stop = start
        while stop + 1 < len(scores) and sorted_scores[stop + 1] == sorted_scores[start]:
            stop += 1
        if stop > start:
            ranks[order[start:stop + 1]] = (start + stop + 2) / 2.0
        start = stop + 1
    return float((ranks[labels].sum() - positives * (positives + 1) / 2.0) / (positives * negatives))


def macro_auc(y_true, probabilities) -> float:
    """Средняя по классам площадь под ROC-кривой (схема «один против остальных»)."""
    y_true = np.asarray(y_true, dtype=int)
    values = [binary_auc(y_true == label, np.asarray(probabilities)[:, label])
              for label in range(np.asarray(probabilities).shape[1])]
    values = [v for v in values if not np.isnan(v)]
    return float(np.mean(values)) if values else float("nan")


def classification_report(y_true, probabilities) -> dict:
    """
    Сводный отчёт о качестве классификации по матрице вероятностей.

    Возвращает словарь из простых типов Python — его удобно хранить в базе
    данных (JSON) и показывать в интерфейсе.
    """
    probabilities = np.asarray(probabilities, dtype=float)
    y_true = np.asarray(y_true, dtype=int)
    y_pred = probabilities.argmax(axis=1)
    n_classes = probabilities.shape[1]
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    scores = per_class_scores(matrix)
    return {
        "n": int(len(y_true)),
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": float(scores["recall"].mean()),
        "macro_f1": float(scores["f1"].mean()),
        "macro_auc": macro_auc(y_true, probabilities),
        "log_loss": log_loss(y_true, probabilities),
        "precision": [float(v) for v in scores["precision"]],
        "recall": [float(v) for v in scores["recall"]],
        "f1": [float(v) for v in scores["f1"]],
        "support": [int(v) for v in scores["support"]],
        "confusion_matrix": matrix.tolist(),
    }


def one_hot_probabilities(labels, n_classes: int) -> np.ndarray:
    """Превращает «жёсткие» ответы в матрицу вероятностей (для наивных моделей)."""
    labels = np.asarray(labels, dtype=int)
    # Небольшое сглаживание, чтобы логарифмическая функция потерь оставалась конечной
    probabilities = np.full((len(labels), n_classes), 0.02 / max(n_classes - 1, 1))
    probabilities[np.arange(len(labels)), labels] = 0.98
    return probabilities
