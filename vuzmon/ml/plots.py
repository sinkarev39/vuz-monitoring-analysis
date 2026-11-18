# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль plots — графики для визуального анализа данных и результатов обучения.

Все графики строятся библиотекой matplotlib без графического окна (режим
«Agg»), подписи выполнены на русском языке. Один и тот же код используется
в веб-интерфейсе (векторные изображения SVG) и в блокнотах Jupyter.
"""

import io

import matplotlib

matplotlib.use("Agg")   # построение без оконной системы — обязательно для сервера

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# Единая палитра системы. Цвета классов различимы и при нарушениях цветового зрения
COLORS = {"primary": "#1f4e79", "accent": "#2a7fb8", "grey": "#8c96a5", "light": "#dfe5ec",
          "text": "#1c2733", "warm": "#c44e52"}
CLASS_COLORS = ("#c44e52", "#9aa5b1", "#2a7fb8")   # зона риска, средний уровень, лидеры

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#aab4c0",
    "axes.labelcolor": COLORS["text"], "xtick.color": "#4a5563", "ytick.color": "#4a5563",
    "axes.grid": True, "grid.color": "#e6ebf0", "grid.linewidth": 0.8, "axes.axisbelow": True,
    "legend.frameon": False, "figure.dpi": 100, "svg.fonttype": "none",
})


def to_svg(figure) -> str:
    """Преобразует рисунок в текст SVG (для вставки в HTML-страницу) и освобождает память."""
    buffer = io.StringIO()
    figure.savefig(buffer, format="svg", bbox_inches="tight")
    plt.close(figure)
    svg = buffer.getvalue()
    return svg[svg.index("<svg"):]


def to_png(figure, path, dpi: int = 200) -> None:
    """Сохраняет рисунок в файл PNG (для отчётов) и освобождает память."""
    figure.savefig(path, format="png", dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def learning_curves(history: list, best_epoch: int = None):
    """Кривые обучения: функция потерь и доля верных ответов по эпохам."""
    epochs = [record["epoch"] for record in history]
    has_metric = any(record.get("val_metric") is not None for record in history)
    figure, axes = plt.subplots(1, 2 if has_metric else 1, figsize=(9.5 if has_metric else 5.5, 3.4))
    axes = np.atleast_1d(axes)
    axes[0].plot(epochs, [r["train_loss"] for r in history], color=COLORS["primary"], label="обучение")
    if any(r.get("val_loss") is not None for r in history):
        axes[0].plot(epochs, [r.get("val_loss") for r in history], color=COLORS["warm"], label="валидация")
    axes[0].set(title="Функция потерь", xlabel="Эпоха")
    if has_metric:
        axes[1].plot(epochs, [r["train_metric"] for r in history], color=COLORS["primary"], label="обучение")
        axes[1].plot(epochs, [r["val_metric"] for r in history], color=COLORS["warm"], label="валидация")
        axes[1].set(title="Доля верных ответов", xlabel="Эпоха")
    for axis in axes:
        if best_epoch:
            axis.axvline(best_epoch, color=COLORS["grey"], linestyle="--", linewidth=1)
            axis.annotate("лучшая эпоха", (best_epoch, axis.get_ylim()[1]), fontsize=8,
                          color="#4a5563", ha="center", va="bottom")
        axis.legend(loc="best")
    figure.tight_layout()
    return figure


def confusion_matrix(matrix, labels, title: str = "Матрица ошибок",
                     xlabel: str = "Предсказанный класс", ylabel: str = "Фактический класс"):
    """Матрица ошибок с числом объектов и долей по строке в каждой клетке."""
    matrix = np.asarray(matrix, dtype=float)
    shares = matrix / np.maximum(matrix.sum(axis=1, keepdims=True), 1)
    figure, axis = plt.subplots(figsize=(4.6, 3.8))
    axis.imshow(shares, cmap="Blues", vmin=0, vmax=1)
    axis.grid(False)
    axis.set_xticks(range(len(labels)), labels, rotation=15)
    axis.set_yticks(range(len(labels)), labels)
    axis.set(xlabel=xlabel, ylabel=ylabel, title=title)
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            color = "white" if shares[row, column] > 0.55 else COLORS["text"]
            axis.text(column, row, f"{int(matrix[row, column])}\n{shares[row, column]:.0%}",
                      ha="center", va="center", color=color, fontsize=10)
    figure.tight_layout()
    return figure


def class_distribution(years, counts, labels, title: str = "Распределение организаций по классам"):
    """Столбцы с накоплением: число организаций каждого класса по годам."""
    counts = np.asarray(counts, dtype=float)     # строки — годы, столбцы — классы
    figure, axis = plt.subplots(figsize=(6.4, 3.4))
    bottom = np.zeros(len(years))
    for index, label in enumerate(labels):
        bars = axis.bar([str(y) for y in years], counts[:, index], bottom=bottom,
                        color=CLASS_COLORS[index], label=label, width=0.62)
        for bar, value in zip(bars, counts[:, index]):
            if value > 0:
                axis.text(bar.get_x() + bar.get_width() / 2, bar.get_y() + value / 2, f"{int(value)}",
                          ha="center", va="center", color="white", fontsize=9)
        bottom += counts[:, index]
    axis.set(title=title, ylabel="Число организаций")
    axis.legend(ncol=len(labels), loc="upper center", bbox_to_anchor=(0.5, -0.12))
    axis.grid(axis="x", visible=False)
    figure.tight_layout()
    return figure


def horizontal_bars(names, values, title: str, xlabel: str = "", color: str = None, errors=None):
    """Горизонтальные столбцы (важность признаков, рейтинги). Первый элемент — сверху."""
    figure, axis = plt.subplots(figsize=(7.2, max(2.2, 0.34 * len(names) + 1.0)))
    positions = np.arange(len(names))[::-1]
    axis.barh(positions, values, xerr=errors, color=color or COLORS["accent"], height=0.66,
              error_kw={"ecolor": COLORS["grey"], "capsize": 2})
    axis.set_yticks(positions, [name if len(name) <= 60 else name[:57] + "…" for name in names])
    axis.set(title=title, xlabel=xlabel)
    axis.grid(axis="y", visible=False)
    figure.tight_layout()
    return figure


def indicator_dynamics(series: list):
    """
    Динамика основных показателей организации в сравнении с медианой по России.
    series — список словарей {'title', 'years', 'values', 'medians'}.
    """
    count = max(len(series), 1)
    figure, axes = plt.subplots(1, count, figsize=(2.75 * count, 2.7), squeeze=False)
    for axis, item in zip(axes[0], series):
        axis.plot(item["years"], item["values"], color=COLORS["primary"], marker="o", markersize=4,
                  label="организация")
        if any(v is not None for v in item["medians"]):
            axis.plot(item["years"], item["medians"], color=COLORS["warm"], linestyle="--", marker="s",
                      markersize=3, label="медиана по РФ")
        axis.set_title(item["title"], fontsize=9)
        axis.set_xticks(item["years"], [str(y)[2:] for y in item["years"]])
        axis.tick_params(labelsize=8)
    handles, labels = axes[0][0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=9)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    return figure


def grouped_bars(groups, series: dict, title: str, ylabel: str = "", reference: float = None):
    """Сгруппированные столбцы: сравнение нескольких организаций или моделей по набору показателей."""
    figure, axis = plt.subplots(figsize=(max(6.0, 1.5 * len(groups) + 2.5), 3.6))
    palette = [COLORS["primary"], COLORS["accent"], COLORS["warm"], COLORS["grey"], "#e0a526", "#5b8c5a"]
    width = 0.8 / max(len(series), 1)
    for index, (name, values) in enumerate(series.items()):
        positions = np.arange(len(groups)) + index * width - 0.4 + width / 2
        axis.bar(positions, values, width=width * 0.92, label=name if len(name) <= 42 else name[:40] + "…",
                 color=palette[index % len(palette)])
    if reference is not None:
        axis.axhline(reference, color=COLORS["text"], linestyle="--", linewidth=1)
    axis.set_xticks(np.arange(len(groups)), groups)
    axis.set(title=title, ylabel=ylabel)
    axis.grid(axis="x", visible=False)
    axis.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=2, fontsize=9)
    figure.tight_layout()
    return figure


def scatter_map(x, y, groups, group_labels, highlight=None, title: str = "", xlabel: str = "", ylabel: str = ""):
    """
    Карта организаций в двумерном пространстве (проекция скрытых представлений).
    groups — номер группы для каждой точки; highlight — маска выделенных точек.
    """
    x, y, groups = np.asarray(x), np.asarray(y), np.asarray(groups)
    palette = [COLORS["primary"], "#e0a526", COLORS["accent"], "#5b8c5a", "#8a6bb0", COLORS["grey"], "#c98a4b"]
    figure, axis = plt.subplots(figsize=(6.6, 4.6))
    for index, label in enumerate(group_labels):
        mask = groups == index
        axis.scatter(x[mask], y[mask], s=16, alpha=0.75, color=palette[index % len(palette)],
                     label=f"{label} ({int(mask.sum())})", linewidths=0)
    if highlight is not None and np.any(highlight):
        axis.scatter(x[highlight], y[highlight], s=70, facecolors="none", edgecolors=COLORS["warm"],
                     linewidths=1.6, label="нетипичные")
    axis.set(title=title, xlabel=xlabel, ylabel=ylabel)
    axis.legend(fontsize=8, loc="best")
    figure.tight_layout()
    return figure


def histogram(values, title: str, xlabel: str, threshold: float = None, threshold_label: str = ""):
    """Гистограмма распределения величины с необязательной отметкой порога."""
    figure, axis = plt.subplots(figsize=(6.0, 3.2))
    axis.hist(np.asarray(values, dtype=float), bins=40, color=COLORS["accent"], alpha=0.9)
    if threshold is not None:
        axis.axvline(threshold, color=COLORS["warm"], linestyle="--")
        axis.annotate(threshold_label, (threshold, axis.get_ylim()[1] * 0.9), color=COLORS["warm"],
                      fontsize=9, ha="left", xytext=(4, 0), textcoords="offset points")
    axis.set(title=title, xlabel=xlabel, ylabel="Число организаций")
    figure.tight_layout()
    return figure


def pca_2d(points: np.ndarray) -> np.ndarray:
    """Проекция точек на две главные компоненты (через сингулярное разложение)."""
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    return centered @ components[:2].T
