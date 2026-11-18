# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Тесты подсистемы машинного обучения: предобработка, метрики, сети, обучение,
сохранение и загрузка моделей, формирование выборки.
"""

import numpy as np
import pandas as pd
import pytest
import torch

from vuzmon.ml import metrics, pipeline, registry, trainer
from vuzmon.ml.dataset import apply_scenario, build_samples, class_from_count, temporal_split
from vuzmon.ml.networks import MonitoringAutoencoder, MonitoringMLP, count_distribution
from vuzmon.ml.preprocessing import TabularPreprocessor
from vuzmon.ml.softmax_numpy import SoftmaxRegressionGD, softmax
from vuzmon.ml.trainer import TrainConfig


# ------------------------------------------------------------ предобработка
def test_preprocessor_log_impute_and_flags(sample_table):
    pre = TabularPreprocessor().fit(sample_table)
    assert pre.log_columns == ["big"]                 # логарифмируется только признак с «тяжёлым хвостом»
    assert pre.flag_columns == ["a"]                  # флаг пропуска — только там, где пропуски есть
    x = pre.transform(sample_table)
    assert x.shape == (12, 4) and x.dtype == np.float32 and np.isfinite(x).all()
    assert x[3, 3] == 1.0 and x[:, 3].sum() == 1.0    # флаг стоит ровно у строки с пропуском
    assert abs(x[:, 0].mean()) < 1e-5                 # после стандартизации среднее равно нулю


def test_preprocessor_state_roundtrip_and_new_columns(sample_table):
    pre = TabularPreprocessor().fit(sample_table)
    clone = TabularPreprocessor.from_state(pre.state_dict())
    np.testing.assert_allclose(pre.transform(sample_table), clone.transform(sample_table))
    # Таблица без одного из столбцов и с выбросом: пропуск заполняется, выброс ограничивается
    other = pd.DataFrame({"big": [10.0 ** 12], "flag": [1.0]})
    x = clone.transform(other)
    assert np.isfinite(x).all() and np.abs(x).max() <= clone.clip


# ------------------------------------------------------------------ метрики
def test_metrics_on_known_example():
    y = [0, 0, 1, 1, 2, 2]
    proba = np.array([[.8, .1, .1], [.2, .7, .1], [.1, .8, .1], [.1, .6, .3], [.1, .2, .7], [.5, .3, .2]])
    report = metrics.classification_report(y, proba)
    assert report["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [1, 0, 1]]
    assert report["accuracy"] == pytest.approx(4 / 6)
    assert report["recall"] == pytest.approx([0.5, 1.0, 0.5])
    assert report["precision"] == pytest.approx([0.5, 2 / 3, 1.0])
    assert report["balanced_accuracy"] == pytest.approx(2 / 3)


def test_auc_matches_sklearn():
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(1)
    y = rng.integers(0, 3, 300)
    proba = softmax(rng.normal(size=(300, 3)) + np.eye(3)[y])
    proba = np.round(proba, 2)                       # округление создаёт совпадающие оценки
    proba /= proba.sum(axis=1, keepdims=True)
    assert metrics.macro_auc(y, proba) == pytest.approx(roc_auc_score(y, proba, multi_class="ovr"), abs=1e-9)


# --------------------------------------------------------------------- сети
def test_count_distribution_is_poisson_binomial():
    p = torch.tensor([[0.5, 0.5], [1.0, 0.0], [0.2, 0.9]])
    result = count_distribution(p).numpy()
    np.testing.assert_allclose(result[0], [0.25, 0.5, 0.25], atol=1e-6)
    np.testing.assert_allclose(result[1], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(result[2], [0.08, 0.74, 0.18], atol=1e-6)
    np.testing.assert_allclose(result.sum(axis=1), 1.0, atol=1e-6)


def test_multitask_network_outputs():
    model = MonitoringMLP(10, 3, (16, 8), dropout=0.1, n_aux=5, structure_weight=0.75).eval()
    x = torch.randn(7, 10)
    logits, aux = model.forward_all(x)
    assert logits.shape == (7, 3) and aux.shape == (7, 5)
    proba = model.class_probabilities(x)
    assert proba.shape == (7, 3) and torch.allclose(proba.sum(dim=1), torch.ones(7), atol=1e-5)
    assert MonitoringMLP(10, 3, (16,), n_aux=0).forward_all(x)[1] is None


def test_autoencoder_shapes():
    model = MonitoringAutoencoder(12, (8,), latent_dim=3).eval()
    x = torch.randn(5, 12)
    assert model(x).shape == (5, 12) and model.encode(x).shape == (5, 3)


# ----------------------------------------------------------------- обучение
def _toy_problem(n=400, seed=0):
    """
    Учебная задача с той же структурой, что и настоящая: пять «показателей»,
    каждый выполнен, если соответствующий признак положителен; класс
    определяется числом выполненных показателей (0–1, 2–3, 4–5).
    """
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 6)).astype(np.float32)
    aux = (x[:, :5] > 0).astype(np.float32)
    y = np.array([class_from_count(count) for count in aux.sum(axis=1)])
    return x, y, aux


def test_softmax_numpy_learns_and_saves(tmp_path):
    x, y, _ = _toy_problem()
    model = SoftmaxRegressionGD(epochs=80, learning_rate=0.1).fit(x[:300], y[:300], x[300:], y[300:])
    assert model.history[-1]["train_loss"] < model.history[0]["train_loss"]
    assert metrics.accuracy(y[300:], model.predict(x[300:])) > 0.7
    model.save(tmp_path / "model.npz")
    restored = SoftmaxRegressionGD.load(tmp_path / "model.npz")
    np.testing.assert_allclose(restored.predict_proba(x[:5]), model.predict_proba(x[:5]))


def test_training_is_reproducible_and_stops_early():
    x, y, aux = _toy_problem()
    config = TrainConfig(hidden=(16,), epochs=150, patience=10, learning_rate=0.01, seed=7)
    first = trainer.train_classifier(x[:300], y[:300], x[300:], y[300:], config, None, aux[:300], aux[300:])
    second = trainer.train_classifier(x[:300], y[:300], x[300:], y[300:], config, None, aux[:300], aux[300:])
    np.testing.assert_allclose(trainer.predict_proba(first.model, x[300:]),
                               trainer.predict_proba(second.model, x[300:]), atol=1e-6)
    assert metrics.accuracy(y[300:], trainer.predict_proba(first.model, x[300:]).argmax(axis=1)) > 0.85
    assert 1 <= first.best_epoch <= len(first.history) <= 150
    assert {"epoch", "train_loss", "val_loss", "train_metric", "val_metric"} <= set(first.history[0])


def test_on_epoch_callback_can_interrupt():
    x, y, _ = _toy_problem(200)
    seen = []
    result = trainer.train_classifier(x[:150], y[:150], x[150:], y[150:], TrainConfig(hidden=(8,), epochs=30, patience=0),
                                      on_epoch=lambda record: seen.append(record["epoch"]) or len(seen) < 3)
    assert seen == [1, 2, 3] and len(result.history) == 3


def test_ensemble_save_load_roundtrip(tmp_path):
    x, y, aux = _toy_problem()
    config = TrainConfig(hidden=(16,), epochs=15, patience=0, ensemble_size=3)
    result = trainer.train_classifier(x[:300], y[:300], x[300:], y[300:], config, None, aux[:300], aux[300:])
    assert len(result.model.members) == 3
    pre = TabularPreprocessor().fit(pd.DataFrame(x, columns=list("abcdef")))
    path = registry.save_classifier(tmp_path / "model1.pt", result.model, pre, config.to_dict(),
                                    {"test": {"accuracy": np.float64(0.9)}}, ("a", "b", "c"), result.history)
    loaded = registry.load_model(path)                # загрузка в безопасном режиме weights_only=True
    assert loaded.kind == "mlp" and loaded.meta["author"] == "Шинкарев Сергей Дмитриевич"
    assert loaded.meta["metrics"]["test"]["accuracy"] == 0.9
    np.testing.assert_allclose(trainer.predict_proba(loaded.model, x[:10]),
                               trainer.predict_proba(result.model, x[:10]), atol=1e-6)
    assert registry.next_model_path(tmp_path).name == "model2.pt"


def test_load_model_rejects_foreign_file(tmp_path):
    torch.save({"weights": torch.zeros(3)}, tmp_path / "other.pt")
    with pytest.raises(ValueError):
        registry.load_model(tmp_path / "other.pt")


# ------------------------------------------------------------------ выборка
def test_class_from_count():
    assert [class_from_count(c) for c in range(6)] == [0, 0, 1, 1, 2, 2]
    assert np.isnan(class_from_count(np.nan))


def test_build_samples_and_temporal_split(small_frames):
    samples = build_samples(small_frames)
    assert (samples.meta["is_branch"] == 0).all()                  # по умолчанию только головные организации
    assert samples.features.shape[0] == len(samples.target) == len(samples.aux_target)
    labeled = samples.labeled()
    assert set(labeled.target.unique()) <= {0.0, 1.0, 2.0}
    # Класс следующего года согласован с числом выполненных показателей
    counts = labeled.aux_target.sum(axis=1).map(class_from_count)
    assert (counts.to_numpy() == labeled.target.to_numpy()).all()
    parts = temporal_split(samples)
    assert parts["train"].meta["year"].max() < parts["val"].meta["year"].min() \
        < parts["test"].meta["year"].min() < parts["future"].meta["year"].min()
    assert parts["future"].target.isna().all()
    # Упорядоченные запасы действительно упорядочены
    ordered = samples.features[[f"margin_sorted_{i}" for i in range(1, 6)]].to_numpy()
    assert (np.diff(ordered, axis=1) >= -1e-9).all()


def test_scenario_changes_only_main_indicator_features(small_frames):
    samples = build_samples(small_frames)
    row = samples.features.iloc[[0]]
    changed = apply_scenario(row, {"e1": 90.0}, {"e1": 60.0}, {"e1": 45.0})
    assert changed["e1_ratio"].iloc[0] == pytest.approx(1.5)
    assert changed["e1_pass"].iloc[0] == 1.0 and changed["e1_growth"].iloc[0] == pytest.approx(1.0)
    untouched = [c for c in row.columns if not c.startswith(("e1_", "margin_sorted", "pass_count"))]
    pd.testing.assert_frame_equal(changed[untouched], row[untouched])


def test_pipeline_end_to_end_on_small_data(small_frames):
    prepared = pipeline.prepare_data(small_frames)
    naive = pipeline.persistence_reports(prepared)
    outcome = pipeline.run_mlp(prepared, TrainConfig(hidden=(32,), epochs=25, patience=8))
    assert outcome["test"]["accuracy"] > 0.5 and naive["test"]["accuracy"] > 0.5
    assert outcome["test"]["macro_auc"] > 0.75        # вероятности модели содержательны
    assert np.isfinite(prepared.x["future"]).all()
