# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль jobs — длительные операции, выполняемые в фоновых потоках:
обучение моделей, подбор гиперпараметров, сбор данных с сайта мониторинга.

Пользователь не ждёт завершения операции: страница сразу возвращается, а ход
работ отображается по данным из базы (метрики эпох) и из реестра задач.
"""

import json
import logging
import threading
import uuid
from datetime import datetime

import joblib
from sqlalchemy import delete

from vuzmon import config
from vuzmon.collect import remote, validation
from vuzmon.collect.crawler import MonitoringCrawler, records_to_frames, save_frames
from vuzmon.collect.http_client import CachedHttpClient
from vuzmon.db.loader import load_monitoring_frames
from vuzmon.db.models import AuditLog, Dataset, Experiment, ExperimentEpoch, Prediction
from vuzmon.db.session import get_number_setting, get_session_factory
from vuzmon.ml import explain, pipeline, registry, trainer
from vuzmon.ml.dataset import CLASS_NAMES
from vuzmon.ml.trainer import TrainConfig
from webapp.services import store

logger = logging.getLogger(__name__)

# Названия видов моделей для интерфейса
MODEL_TYPES = {
    "mlp": "Нейронная сеть (многослойный перцептрон)",
    "autoencoder": "Автокодировщик (профили и нетипичные записи)",
    "softmax": "Softmax-регрессия на NumPy (опорная модель)",
    "forest": "Случайный лес (scikit-learn)",
    "boosting": "Градиентный бустинг (scikit-learn)",
    "search": "Подбор гиперпараметров нейронной сети",
}

# Реестр фоновых задач: идентификатор -> состояние
_jobs = {}
_jobs_lock = threading.Lock()


def job_state(job_id: str) -> dict:
    """Текущее состояние фоновой задачи (или пустой словарь, если такой задачи нет)."""
    with _jobs_lock:
        return dict(_jobs.get(job_id, {}))


def latest_job(kind: str) -> dict:
    """Последняя по времени задача указанного вида."""
    with _jobs_lock:
        candidates = [dict(job, id=job_id) for job_id, job in _jobs.items() if job["kind"] == kind]
    return max(candidates, key=lambda job: job["started"]) if candidates else {}


def _start(kind: str, target, *args) -> str:
    """Запускает функцию target в фоновом потоке и возвращает идентификатор задачи."""
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {"kind": kind, "status": "running", "message": "Задача запущена",
                         "progress": 0.0, "started": datetime.now()}

    def report(message: str, progress: float = None) -> None:
        with _jobs_lock:
            _jobs[job_id]["message"] = message
            if progress is not None:
                _jobs[job_id]["progress"] = progress

    def runner() -> None:
        try:
            target(report, *args)
            with _jobs_lock:
                _jobs[job_id].update(status="done", progress=1.0)
        except Exception as error:
            logger.exception("Фоновая задача %s завершилась ошибкой", kind)
            with _jobs_lock:
                _jobs[job_id].update(status="failed", message=f"Ошибка: {error}")
        finally:
            get_session_factory().remove()

    threading.Thread(target=runner, name=f"job-{kind}-{job_id}", daemon=True).start()
    return job_id


# --------------------------------------------------------------------------
# Обучение моделей
# --------------------------------------------------------------------------
def _train_mlp(db, experiment, prepared, params: dict) -> dict:
    config_ = TrainConfig.from_dict(params)

    def on_epoch(record: dict):
        # Метрики эпохи сразу попадают в базу — страница эксперимента рисует по ним кривые
        db.add(ExperimentEpoch(experiment_id=experiment.id, epoch=record["epoch"],
                               train_loss=record["train_loss"], val_loss=record["val_loss"],
                               train_metric=record["train_metric"], val_metric=record["val_metric"],
                               learning_rate=record["learning_rate"]))
        db.commit()

    outcome = pipeline.run_mlp(prepared, config_, on_epoch)
    result = outcome["result"]
    importance = explain.permutation_importance(
        lambda x: trainer.predict_proba(result.model, x), prepared.x["val"], prepared.y["val"],
        prepared.feature_names, n_repeats=3)[:20]
    metrics_ = {"val": outcome["val"], "test": outcome["test"],
                "persistence": pipeline.persistence_reports(prepared),
                "best_epoch": result.best_epoch, "epochs_run": len(result.history),
                "stopped_early": result.stopped_early, "n_parameters": result.model.count_parameters(),
                "data": prepared.summary(), "importance": importance}
    path = registry.next_model_path(config.MODELS_DIR, ".pt")
    registry.save_classifier(path, result.model, prepared.preprocessor, config_.to_dict(), metrics_,
                             CLASS_NAMES, result.history)
    return {"metrics": metrics_, "path": path}


def _train_autoencoder(db, experiment, prepared, params: dict) -> dict:
    def on_epoch(record: dict):
        db.add(ExperimentEpoch(experiment_id=experiment.id, epoch=record["epoch"],
                               train_loss=record["train_loss"], val_loss=record["val_loss"],
                               learning_rate=record["learning_rate"]))
        db.commit()

    outcome = pipeline.fit_autoencoder(
        prepared, hidden=tuple(params.get("hidden", (64, 24))), latent_dim=int(params.get("latent_dim", 8)),
        epochs=int(params.get("epochs", 150)), n_clusters=int(params.get("n_clusters", 5)),
        anomaly_quantile=float(params.get("anomaly_quantile", 0.95)), seed=int(params.get("seed", 42)),
        on_epoch=on_epoch)
    result = outcome["result"]
    metrics_ = {"val_loss": outcome["val_loss"], "threshold": outcome["threshold"],
                "train_error_mean": outcome["train_error_mean"], "best_epoch": result.best_epoch,
                "epochs_run": len(result.history), "n_parameters": result.model.count_parameters(),
                "n_clusters": len(outcome["centers"]), "data": prepared.summary()}
    path = registry.next_model_path(config.MODELS_DIR, ".pt")
    registry.save_autoencoder(path, result.model, prepared.preprocessor, metrics_, result.history,
                              {"threshold": outcome["threshold"], "centers": outcome["centers"]})
    return {"metrics": metrics_, "path": path}


def _train_classic(db, experiment, prepared, params: dict) -> dict:
    """Softmax-регрессия на NumPy и модели scikit-learn (для сравнения с нейросетью)."""
    kind = experiment.model_type
    if kind == "softmax":
        outcome = pipeline.run_softmax(
            prepared, learning_rate=float(params.get("learning_rate", 0.05)),
            epochs=int(params.get("epochs", 200)), l2=float(params.get("weight_decay", 1e-3)),
            batch_size=int(params.get("batch_size", 128)), seed=int(params.get("seed", 42)))
        for record in outcome["model"].history:
            db.add(ExperimentEpoch(experiment_id=experiment.id, epoch=record["epoch"],
                                   train_loss=record["train_loss"], val_loss=record.get("val_loss")))
        path = registry.next_model_path(config.MODELS_DIR, ".npz")
        outcome["model"].save(path)
    else:
        outcome = pipeline.run_sklearn(prepared, kind, seed=int(params.get("seed", 42)))
        path = registry.next_model_path(config.MODELS_DIR, ".joblib")
        joblib.dump(outcome["model"], path)
    metrics_ = {"val": outcome["val"], "test": outcome["test"],
                "persistence": pipeline.persistence_reports(prepared), "data": prepared.summary()}
    return {"metrics": metrics_, "path": path}


def _run_search(db, experiment, prepared, params: dict, report) -> dict:
    """Случайный поиск гиперпараметров; лучшая конфигурация переобучается и сохраняется."""
    n_trials = int(params.get("n_trials", 15))
    done = []

    def on_trial(trial: dict):
        done.append(trial)
        report(f"Испытание {len(done)} из {n_trials}: F1 = {trial['val_macro_f1']:.3f}", len(done) / (n_trials + 1))
        experiment.message = f"Выполнено испытаний: {len(done)} из {n_trials}"
        db.commit()

    trials = pipeline.random_search(prepared, n_trials=n_trials, seed=int(params.get("seed", 42)),
                                    epochs=int(params.get("epochs", 120)), on_trial=on_trial)
    best = trials[0]["params"]
    report("Обучение лучшей конфигурации", n_trials / (n_trials + 1))
    final = _train_mlp(db, experiment, prepared, best)
    final["metrics"]["trials"] = trials
    experiment.params_json = json.dumps({**params, "best": best}, ensure_ascii=False)
    return final


def _training_job(report, experiment_id: int) -> None:
    db = get_session_factory()()
    experiment = db.get(Experiment, experiment_id)
    try:
        experiment.status = "running"
        db.execute(delete(ExperimentEpoch).where(ExperimentEpoch.experiment_id == experiment_id))
        db.commit()
        report("Подготовка данных", 0.05)
        prepared = store.prepared(experiment.dataset.location)
        params = json.loads(experiment.params_json or "{}")
        report("Обучение модели", 0.1)
        if experiment.model_type == "mlp":
            outcome = _train_mlp(db, experiment, prepared, params)
        elif experiment.model_type == "autoencoder":
            outcome = _train_autoencoder(db, experiment, prepared, params)
        elif experiment.model_type == "search":
            outcome = _run_search(db, experiment, prepared, params, report)
        else:
            outcome = _train_classic(db, experiment, prepared, params)
        experiment.metrics_json = json.dumps(outcome["metrics"], ensure_ascii=False)
        experiment.model_file = store.relative(outcome["path"])
        experiment.status, experiment.message = "done", "Обучение завершено"
        # Первая обученная модель своего вида сразу становится рабочей
        kind = "mlp" if experiment.model_type == "search" else experiment.model_type
        if kind in ("mlp", "autoencoder") and store.active_experiment(db, kind) is None:
            experiment.is_active = True
    except Exception as error:
        db.rollback()
        experiment = db.get(Experiment, experiment_id)
        experiment.status = "failed"
        # На странице эксперимента — краткая причина; полная трассировка пишется в журнал сервера
        experiment.message = f"Обучение не выполнено. Причина: {type(error).__name__}: {error}"[:1000]
        raise
    finally:
        experiment.finished_at = datetime.now()
        db.add(AuditLog(user_id=experiment.created_by, action="train_finish",
                        details=f"Эксперимент №{experiment.id} «{experiment.name}»: {experiment.status}"))
        db.commit()


def start_training(experiment_id: int) -> str:
    """Ставит обучение эксперимента в фоновую очередь."""
    return _start("train", _training_job, experiment_id)


# --------------------------------------------------------------------------
# Сбор данных с сайта мониторинга
# --------------------------------------------------------------------------
def _collect_job(report, years: list, user_id: int, load_to_db: bool, limit: int = None) -> None:
    db = get_session_factory()()
    interval = get_number_setting(db, "collector_interval", 0.25, 0.1, 10.0)
    client = CachedHttpClient(config.CACHE_DIR, min_interval=interval)
    state = {"year_index": 0}

    def progress(done: int, total: int, message: str) -> None:
        share = (state["year_index"] + done / max(total, 1)) / len(years)
        report(f"Сбор данных: {message}", 0.9 * share)

    crawler = MonitoringCrawler(client, progress=progress)
    records, errors = [], []
    for index, year in enumerate(years):
        state["year_index"] = index
        report(f"Сбор данных за {year} год", 0.9 * index / len(years))
        part = crawler.crawl_year(year, limit=limit)
        records.extend(part.records)
        errors.extend(part.errors)
    if not records:
        raise RuntimeError("Не удалось получить ни одной страницы организации")

    report("Сохранение набора данных", 0.92)
    frames = records_to_frames(records)
    directory = store.new_dataset_directory("site")
    save_frames(frames, directory)
    dataset = Dataset(name=f"Сбор с сайта мониторинга от {datetime.now():%d.%m.%Y %H:%M}", source="site",
                      location=store.relative(directory), years=", ".join(map(str, sorted(years))),
                      rows_count=len(frames["institutions"]), created_by=user_id,
                      description=f"Скачано страниц: {client.stats['downloaded']}, из кэша: "
                                  f"{client.stats['from_cache']}, ошибок: {len(errors)}")
    db.add(dataset)
    db.commit()
    note = ""
    if load_to_db:
        # Пробный сбор (один год, несколько организаций) рабочим набором стать не может
        problem = store.activation_problem(dataset.location)
        if problem:
            note = f". Набор сохранён, но рабочим не сделан. {problem}"
        else:
            report("Загрузка данных в базу", 0.95)
            load_monitoring_frames(db, frames)
            store.set_active_dataset(db, dataset.id)
    db.add(AuditLog(user_id=user_id, action="collect_finish",
                    details=f"Набор данных №{dataset.id}: организаций {len(frames['institutions'])}, "
                            f"ошибок {len(errors)}{note}"))
    db.commit()
    report(f"Готово: записей об организациях — {len(frames['institutions'])}, ошибок — {len(errors)}{note}", 1.0)


def start_collect(years: list, user_id: int, load_to_db: bool = True, limit: int = None) -> str:
    """Запускает сбор данных с сайта мониторинга в фоне."""
    return _start("collect", _collect_job, years, user_id, load_to_db, limit)


# --------------------------------------------------------------------------
# Загрузка набора данных по ссылке
# --------------------------------------------------------------------------
def _download_job(report, url: str, user_id: int, name: str = "") -> None:
    """
    Скачивает набор данных по ссылке, проверяет его и добавляет в перечень наборов.
    Рабочим такой набор не становится: это отдельное действие в разделе «Данные».
    """
    db = get_session_factory()()
    frames = remote.load_frames_from_url(
        url, progress=lambda done, total, message: report(f"Загрузка по ссылке: {message}", 0.9 * done / max(total, 1)))
    report("Проверка набора данных", 0.92)
    problems = validation.check_frames(frames)
    if problems:
        raise ValueError("в файлах найдены ошибки. " + ". ".join(problems[:3]))
    directory = store.new_dataset_directory("link")
    save_frames(frames, directory)
    years = sorted(int(year) for year in frames["institutions"]["year"].unique())
    dataset = Dataset(name=(name or f"Загрузка по ссылке от {datetime.now():%d.%m.%Y %H:%M}")[:128], source="link",
                      location=store.relative(directory), years=", ".join(map(str, years)),
                      rows_count=len(frames["institutions"]), created_by=user_id,
                      description=f"Загружено по ссылке: {remote.normalize_url(url)}"[:1000])
    db.add(dataset)
    db.commit()
    db.add(AuditLog(user_id=user_id, action="dataset_link",
                    details=f"Набор данных №{dataset.id} «{dataset.name}» загружен по ссылке {remote.normalize_url(url)}"))
    db.commit()
    report(f"Готово: набор данных №{dataset.id} загружен по ссылке, записей об организациях — {dataset.rows_count}", 1.0)


def start_download(url: str, user_id: int, name: str = "") -> str:
    """
    Запускает загрузку набора данных по ссылке в фоне. Задача относится к тому же
    виду, что и сбор с сайта: одновременно выполняется одна операция получения данных.
    """
    return _start("collect", _download_job, url, user_id, name)


# --------------------------------------------------------------------------
# Построение прогноза (выполняется быстро, поэтому без фонового потока)
# --------------------------------------------------------------------------
def build_forecast(db, user_id: int = None) -> dict:
    """
    Применяет активную нейронную сеть к данным последнего года мониторинга и
    сохраняет прогноз в базу. Если есть активный автокодировщик, к прогнозу
    добавляются оценка нетипичности и номер типологической группы.
    """
    experiment, loaded = store.active_model(db, "mlp")
    if loaded is None:
        raise RuntimeError("Нет активной модели прогноза. Обучите или загрузите модель в разделе «Модели».")
    dataset = store.active_dataset(db)
    if dataset is None:
        raise RuntimeError("Нет набора данных. Загрузите данные в разделе «Данные».")
    future = store.prepared(dataset.location).parts["future"]
    if len(future) == 0:
        raise RuntimeError("В наборе данных нет организаций для прогноза.")
    table = pipeline.forecast(loaded, future)

    _, autoencoder = store.active_model(db, "autoencoder")
    if autoencoder is not None:
        profile = pipeline.profile(autoencoder, future)
        table["anomaly_score"] = profile["anomaly_score"].to_numpy()
        table["cluster"] = profile["cluster"].to_numpy() if "cluster" in profile else None

    base_year = int(table["year"].iloc[0])
    db.execute(delete(Prediction).where(Prediction.base_year == base_year))
    rows = []
    for row in table.itertuples():
        probabilities = [float(getattr(row, f"proba_{i}")) for i in range(len(CLASS_NAMES))]
        rows.append(Prediction(
            experiment_id=experiment.id, university_id=int(row.inst_id), base_year=base_year,
            target_year=base_year + 1, predicted_class=int(row.predicted_class),
            probabilities_json=json.dumps([round(p, 4) for p in probabilities]),
            anomaly_score=float(row.anomaly_score) if "anomaly_score" in table else None,
            cluster=int(row.cluster) if "cluster" in table and row.cluster is not None else None,
            created_by=user_id))
    db.add_all(rows)
    db.commit()
    return {"count": len(rows), "base_year": base_year, "experiment_id": experiment.id}
