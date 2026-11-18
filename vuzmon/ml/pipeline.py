# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль pipeline — сквозной конвейер интеллектуального анализа.

Здесь собраны операции верхнего уровня, которыми пользуются и веб-приложение,
и блокноты Jupyter:
  подготовка данных -> обучение моделей -> оценка качества -> сравнение ->
  подбор гиперпараметров -> прогноз на следующий год -> поиск нетипичных
  организаций и построение типологических групп.
"""

import itertools
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vuzmon.ml import metrics, trainer
from vuzmon.ml.dataset import CLASS_NAMES, SampleSet, build_samples, temporal_split
from vuzmon.ml.preprocessing import TabularPreprocessor
from vuzmon.ml.registry import LoadedModel
from vuzmon.ml.softmax_numpy import SoftmaxRegressionGD
from vuzmon.ml.trainer import TrainConfig

N_CLASSES = len(CLASS_NAMES)


@dataclass
class PreparedData:
    """Выборка, разбитая по времени и переведённая в числовые матрицы."""
    samples: SampleSet
    parts: dict                      # 'train' / 'val' / 'test' / 'future' -> SampleSet
    preprocessor: TabularPreprocessor
    x: dict                          # часть -> матрица признаков float32
    y: dict                          # часть -> вектор классов
    feature_names: list
    aux: dict = None                 # часть -> матрица вспомогательных целей (выполнение показателей)

    def summary(self) -> dict:
        """Краткая сводка о размерах частей и распределении классов."""
        info = {"n_features": len(self.feature_names)}
        for name, part in self.parts.items():
            years = sorted(int(v) for v in part.meta["year"].unique())
            info[name] = {"rows": len(part), "years": years}
            if name != "future":
                counts = np.bincount(self.y[name], minlength=N_CLASSES)
                info[name]["classes"] = [int(v) for v in counts]
        return info


def prepare_data(frames: dict, include_branches: bool = False) -> PreparedData:
    """Строит выборку, делит её по времени и обучает предобработчик на обучающей части."""
    samples = build_samples(frames, include_branches=include_branches)
    parts = temporal_split(samples)
    preprocessor = TabularPreprocessor().fit(parts["train"].features)
    x = {name: preprocessor.transform(part.features) for name, part in parts.items()}
    y = {name: part.target.to_numpy(dtype=np.int64) for name, part in parts.items() if name != "future"}
    aux = {name: part.aux_target.to_numpy(dtype=np.float32) for name, part in parts.items() if name != "future"}
    return PreparedData(samples, parts, preprocessor, x, y, preprocessor.feature_names, aux)


# --------------------------------------------------------------------------
# Обучение и оценка моделей
# --------------------------------------------------------------------------
def persistence_reports(prepared: PreparedData) -> dict:
    """
    Наивный прогноз «в следующем году класс не изменится». Это нижняя планка:
    модель полезна, только если предсказывает лучше.
    """
    reports = {}
    for name in ("val", "test"):
        current = prepared.parts[name].meta["current_class"].fillna(1).to_numpy(dtype=int)
        reports[name] = metrics.classification_report(
            prepared.y[name], metrics.one_hot_probabilities(current, N_CLASSES))
    return reports


def evaluate(predict, prepared: PreparedData) -> dict:
    """Отчёты о качестве на валидационной и тестовой частях для функции predict."""
    return {name: metrics.classification_report(prepared.y[name], predict(prepared.x[name]))
            for name in ("val", "test")}


def run_mlp(prepared: PreparedData, config: TrainConfig, on_epoch=None) -> dict:
    """Обучает нейронную сеть и оценивает её. Возвращает модель, историю и отчёты."""
    result = trainer.train_classifier(prepared.x["train"], prepared.y["train"],
                                      prepared.x["val"], prepared.y["val"], config, on_epoch,
                                      prepared.aux["train"], prepared.aux["val"])
    reports = evaluate(lambda x: trainer.predict_proba(result.model, x), prepared)
    return {"result": result, "config": config, "val": reports["val"], "test": reports["test"]}


def run_softmax(prepared: PreparedData, **params) -> dict:
    """Обучает собственную softmax-регрессию на NumPy (опорная модель)."""
    model = SoftmaxRegressionGD(n_classes=N_CLASSES, **params)
    model.fit(prepared.x["train"], prepared.y["train"], prepared.x["val"], prepared.y["val"])
    reports = evaluate(model.predict_proba, prepared)
    return {"model": model, "val": reports["val"], "test": reports["test"]}


def run_sklearn(prepared: PreparedData, kind: str = "boosting", seed: int = 42) -> dict:
    """Классические модели scikit-learn для сравнения: случайный лес и градиентный бустинг."""
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    if kind == "forest":
        model = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, n_jobs=-1, random_state=seed)
    else:
        model = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_depth=4,
                                               l2_regularization=1.0, early_stopping=False,
                                               random_state=seed)
    model.fit(prepared.x["train"], prepared.y["train"])
    reports = evaluate(model.predict_proba, prepared)
    return {"model": model, "val": reports["val"], "test": reports["test"]}


# --------------------------------------------------------------------------
# Подбор гиперпараметров нейронной сети
# --------------------------------------------------------------------------
SEARCH_SPACE = {
    "hidden": [(64, 32), (128, 64), (256, 128), (256, 128, 64), (128, 64, 32)],
    "dropout": [0.1, 0.2, 0.3, 0.4, 0.5],
    "learning_rate": [3e-4, 1e-3, 3e-3],
    "weight_decay": [1e-5, 1e-4, 1e-3, 1e-2],
    "batch_size": [32, 64, 128],
    "label_smoothing": [0.0, 0.05, 0.1],
    "activation": ["relu", "gelu"],
    "aux_weight": [0.0, 1.0, 2.0, 3.0],
    "structure_weight": [0.5, 0.75, 1.0],
}


def random_search(prepared: PreparedData, n_trials: int = 20, seed: int = 42, epochs: int = 120,
                  on_trial=None) -> list:
    """
    Случайный поиск по сетке гиперпараметров. Качество каждой конфигурации
    измеряется на валидационной части (тестовая часть в подборе не участвует).

    Возвращает список испытаний, отсортированный по убыванию macro-F1.
    """
    rng = np.random.default_rng(seed)
    keys = list(SEARCH_SPACE)
    combinations = list(itertools.product(*(range(len(SEARCH_SPACE[key])) for key in keys)))
    chosen = rng.choice(len(combinations), size=min(n_trials, len(combinations)), replace=False)
    trials = []
    for number, index in enumerate(chosen, start=1):
        params = {key: SEARCH_SPACE[key][position] for key, position in zip(keys, combinations[index])}
        config = TrainConfig(epochs=epochs, patience=15, seed=seed, **params)
        outcome = run_mlp(prepared, config)
        trial = {"trial": number, "params": config.to_dict(), "best_epoch": outcome["result"].best_epoch,
                 "val_accuracy": outcome["val"]["accuracy"], "val_macro_f1": outcome["val"]["macro_f1"],
                 "val_log_loss": outcome["val"]["log_loss"]}
        trials.append(trial)
        if on_trial is not None:
            on_trial(trial)
    return sorted(trials, key=lambda t: (t["val_macro_f1"], -t["val_log_loss"]), reverse=True)


# --------------------------------------------------------------------------
# Применение моделей
# --------------------------------------------------------------------------
def forecast(loaded: LoadedModel, samples: SampleSet) -> pd.DataFrame:
    """
    Применяет классификатор к выборке и возвращает таблицу прогнозов:
    сведения об организации, вероятности классов и предсказанный класс.
    """
    x = loaded.preprocessor.transform(samples.features)
    probabilities = trainer.predict_proba(loaded.model, x)
    table = samples.meta[["inst_id", "year", "name", "region", "current_class"]].copy()
    for index in range(probabilities.shape[1]):
        table[f"proba_{index}"] = probabilities[:, index]
    table["predicted_class"] = probabilities.argmax(axis=1)
    table["target_year"] = table["year"] + 1
    return table


def fit_autoencoder(prepared: PreparedData, hidden=(64, 24), latent_dim: int = 8, epochs: int = 150,
                    n_clusters: int = 5, anomaly_quantile: float = 0.95, seed: int = 42,
                    on_epoch=None) -> dict:
    """
    Обучает автокодировщик на всех организациях обучающего периода, затем:
      * определяет порог нетипичности как квантиль ошибки восстановления;
      * строит типологические группы методом k-средних в скрытом пространстве.
    """
    from sklearn.cluster import KMeans
    result = trainer.train_autoencoder(prepared.x["train"], prepared.x["val"], hidden, latent_dim,
                                       epochs=epochs, seed=seed, on_epoch=on_epoch)
    train_errors = trainer.reconstruction_error(result.model, prepared.x["train"])
    threshold = float(np.quantile(train_errors, anomaly_quantile))
    embeddings = trainer.encode(result.model, prepared.x["train"])
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit(embeddings)
    return {"result": result, "threshold": threshold, "centers": kmeans.cluster_centers_.tolist(),
            "val_loss": result.history[result.best_epoch - 1]["val_loss"] if result.best_epoch else None,
            "train_error_mean": float(train_errors.mean())}


def profile(loaded: LoadedModel, samples: SampleSet) -> pd.DataFrame:
    """
    Применяет автокодировщик: для каждой организации возвращает ошибку
    восстановления, признак нетипичности, номер типологической группы и
    координаты в скрытом пространстве.
    """
    x = loaded.preprocessor.transform(samples.features)
    errors = trainer.reconstruction_error(loaded.model, x)
    embeddings = trainer.encode(loaded.model, x)
    centers = np.asarray(loaded.meta.get("centers", []), dtype=np.float64)
    table = samples.meta[["inst_id", "year", "name", "region"]].copy()
    table["anomaly_score"] = errors
    table["is_anomaly"] = errors > float(loaded.meta.get("threshold", np.inf))
    if centers.size:
        # Номер группы — ближайший центр кластера (евклидово расстояние)
        distances = ((embeddings[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
        table["cluster"] = distances.argmin(axis=1)
    for index in range(embeddings.shape[1]):
        table[f"z{index}"] = embeddings[:, index]
    return table


def top_error_features(loaded: LoadedModel, features_row: pd.DataFrame, top: int = 5) -> list:
    """Признаки, которые автокодировщик восстановил хуже всего (причина нетипичности)."""
    import torch
    x = loaded.preprocessor.transform(features_row)
    with torch.no_grad():
        restored = loaded.model(torch.as_tensor(x, dtype=torch.float32)).numpy()
    errors = (restored[0] - x[0]) ** 2
    names = loaded.preprocessor.feature_names
    order = np.argsort(-errors)[:top]
    return [{"feature": names[i], "error": float(errors[i]), "value": float(x[0, i])} for i in order]
