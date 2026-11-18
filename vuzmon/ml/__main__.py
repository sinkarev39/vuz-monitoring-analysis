# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Обучение всех моделей системы из командной строки:

    python -m vuzmon.ml --trials 30

Сценарий воспроизводит путь, описанный в отчёте:
  1) базовая однозадачная сеть (точка отсчёта);
  2) подбор гиперпараметров многозадачной сети; лучшая конфигурация
     обучается как ансамбль из пяти сетей и сохраняется в model1.pt;
  3) автокодировщик для профилей организаций — model2.pt;
  4) softmax-регрессия на NumPy — model3.npz;
  5) случайный лес (scikit-learn) — model4.joblib.
Сводка метрик записывается в models/metrics.json.
"""

import argparse
import json
import logging
import sys
import time

import joblib

from vuzmon import config
from vuzmon.collect.crawler import load_frames
from vuzmon.ml import explain, pipeline, registry, trainer
from vuzmon.ml.dataset import CLASS_NAMES
from vuzmon.ml.trainer import TrainConfig

# Базовая сеть: один выход, без регуляризации и нормализации, обычный SGD
BASELINE_CONFIG = TrainConfig(hidden=(64, 32), dropout=0.0, batch_norm=False, optimizer="sgd",
                              learning_rate=0.01, weight_decay=0.0, epochs=100, patience=0,
                              scheduler="none", aux_weight=0.0)


def _brief(report: dict) -> dict:
    """Короткая выжимка отчёта о качестве для сводной таблицы."""
    return {key: round(report[key], 4) for key in ("accuracy", "balanced_accuracy", "macro_f1", "macro_auc", "log_loss")}


def train_all(data_dir=config.MONITORING_DIR, models_dir=config.MODELS_DIR, n_trials: int = 30,
              seed: int = config.RANDOM_SEED, ensemble_size: int = 5, log=print) -> dict:
    """Обучает и сохраняет все модели. Возвращает сводку метрик."""
    frames = load_frames(data_dir)
    prepared = pipeline.prepare_data(frames)
    summary = {"data": prepared.summary(), "models": {}}
    naive = pipeline.persistence_reports(prepared)
    summary["models"]["Наивный прогноз (класс не изменится)"] = {"val": _brief(naive["val"]), "test": _brief(naive["test"])}

    log("1/5 Базовая однозадачная сеть")
    base = pipeline.run_mlp(prepared, BASELINE_CONFIG)
    summary["models"]["Базовая сеть (64-32, SGD)"] = {"val": _brief(base["val"]), "test": _brief(base["test"])}
    summary["baseline_history"] = base["result"].history

    log(f"2/5 Подбор гиперпараметров многозадачной сети: испытаний — {n_trials}")
    trials = pipeline.random_search(
        prepared, n_trials=n_trials, seed=seed,
        on_trial=lambda t: log(f"    испытание {t['trial']:2d}: macro-F1 = {t['val_macro_f1']:.3f}, "
                               f"верных ответов = {t['val_accuracy']:.3f}"))
    # Лучшая конфигурация обучается как ансамбль из нескольких сетей: это
    # снижает зависимость результата от случайной инициализации весов
    best_config = TrainConfig.from_dict({**trials[0]["params"], "ensemble_size": ensemble_size})
    single = pipeline.run_mlp(prepared, TrainConfig.from_dict(trials[0]["params"]))
    summary["models"]["Многозадачная сеть, одна (лучшая конфигурация)"] = {
        "val": _brief(single["val"]), "test": _brief(single["test"])}
    log(f"    обучение ансамбля из {ensemble_size} сетей с лучшей конфигурацией")
    best = pipeline.run_mlp(prepared, best_config)
    result = best["result"]
    importance = explain.permutation_importance(
        lambda x: trainer.predict_proba(result.model, x), prepared.x["val"], prepared.y["val"],
        prepared.feature_names, n_repeats=5, seed=seed)[:20]
    metrics = {"val": best["val"], "test": best["test"], "persistence": naive, "best_epoch": result.best_epoch,
               "epochs_run": len(result.history), "stopped_early": result.stopped_early,
               "n_parameters": result.model.count_parameters(), "data": prepared.summary(),
               "importance": importance, "trials": trials}
    registry.save_classifier(models_dir / "model1.pt", result.model, prepared.preprocessor,
                             best_config.to_dict(), metrics, CLASS_NAMES, result.history)
    summary["models"]["Ансамбль многозадачных сетей (model1.pt)"] = {"val": _brief(best["val"]), "test": _brief(best["test"])}
    summary["best_config"] = best_config.to_dict()
    summary["trials"] = trials

    log("3/5 Автокодировщик")
    auto = pipeline.fit_autoencoder(prepared, seed=seed)
    auto_metrics = {"val_loss": auto["val_loss"], "threshold": auto["threshold"],
                    "train_error_mean": auto["train_error_mean"], "best_epoch": auto["result"].best_epoch,
                    "epochs_run": len(auto["result"].history), "n_clusters": len(auto["centers"]),
                    "n_parameters": auto["result"].model.count_parameters(), "data": prepared.summary()}
    registry.save_autoencoder(models_dir / "model2.pt", auto["result"].model, prepared.preprocessor,
                              auto_metrics, auto["result"].history,
                              {"threshold": auto["threshold"], "centers": auto["centers"]})
    summary["autoencoder"] = {key: value for key, value in auto_metrics.items() if key != "data"}

    log("4/5 Softmax-регрессия на NumPy")
    softmax = pipeline.run_softmax(prepared, learning_rate=0.05, epochs=200, seed=seed)
    softmax["model"].save(models_dir / "model3.npz")
    summary["models"]["Softmax-регрессия на NumPy (model3.npz)"] = {"val": _brief(softmax["val"]), "test": _brief(softmax["test"])}

    log("5/5 Случайный лес")
    forest = pipeline.run_sklearn(prepared, "forest", seed=seed)
    joblib.dump(forest["model"], models_dir / "model4.joblib", compress=3)
    summary["models"]["Случайный лес (model4.joblib)"] = {"val": _brief(forest["val"]), "test": _brief(forest["test"])}
    boosting = pipeline.run_sklearn(prepared, "boosting", seed=seed)
    summary["models"]["Градиентный бустинг"] = {"val": _brief(boosting["val"]), "test": _brief(boosting["test"])}

    # Полные отчёты для моделей сравнения: по ним система создаёт записи об экспериментах
    data_summary = prepared.summary()
    summary["reports"] = {
        "model3.npz": {"type": "softmax", "name": "Softmax-регрессия на NumPy (опорная модель)",
                       "params": {"learning_rate": 0.05, "epochs": 200, "weight_decay": 1e-3, "batch_size": 128, "seed": seed},
                       "metrics": {"val": softmax["val"], "test": softmax["test"], "persistence": naive, "data": data_summary},
                       "history": softmax["model"].history},
        "model4.joblib": {"type": "forest", "name": "Случайный лес (scikit-learn)",
                          "params": {"n_estimators": 400, "min_samples_leaf": 2, "seed": seed},
                          "metrics": {"val": forest["val"], "test": forest["test"], "persistence": naive, "data": data_summary},
                          "history": []},
    }
    (models_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    return summary


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Обучение моделей системы анализа данных мониторинга вузов")
    parser.add_argument("--trials", type=int, default=30, help="число испытаний при подборе гиперпараметров")
    parser.add_argument("--data", default=str(config.MONITORING_DIR), help="каталог набора данных")
    parser.add_argument("--models", default=str(config.MODELS_DIR), help="каталог для файлов моделей")
    parser.add_argument("--seed", type=int, default=config.RANDOM_SEED)
    parser.add_argument("--ensemble", type=int, default=5, help="число сетей в итоговом ансамбле")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    config.ensure_dirs()
    started = time.time()
    from pathlib import Path
    summary = train_all(Path(args.data), Path(args.models), args.trials, args.seed, args.ensemble,
                        log=logging.info)
    logging.info("Готово за %.0f с", time.time() - started)
    logging.info("%-44s %10s %10s %10s %10s", "Модель", "верных", "macro-F1", "ROC-AUC", "log-loss")
    for name, parts in summary["models"].items():
        test = parts["test"]
        logging.info("%-44s %10.3f %10.3f %10.3f %10.3f", name, test["accuracy"], test["macro_f1"],
                     test["macro_auc"], test["log_loss"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
