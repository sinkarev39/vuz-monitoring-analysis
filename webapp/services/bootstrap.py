# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль bootstrap — подготовка системы к работе при первом запуске.

Если база данных пуста, в неё загружается набор данных из каталога
data/monitoring, а файлы моделей из каталога models регистрируются как
готовые эксперименты. Благодаря этому система работоспособна сразу после
установки: проверяющему не нужно ничего собирать и обучать заново.
"""

import json
import logging
from datetime import datetime

from sqlalchemy import func, select

from vuzmon import config
from vuzmon.db.loader import load_monitoring_frames
from vuzmon.db.models import AuditLog, Dataset, Experiment, ExperimentEpoch, Prediction, Setting, University
from vuzmon.db.session import get_session
from vuzmon.ml import registry
from webapp.services import store

logger = logging.getLogger(__name__)


INTERRUPTED_MESSAGE = ("Обучение прервано: система была остановлена до его завершения. "
                       "Запустите обучение заново.")


def close_interrupted_experiments() -> int:
    """
    Закрывает эксперименты, обучение которых оборвалось при остановке системы.

    Обучение идёт в фоновом потоке и не переживает перезапуск. Запись
    об эксперименте при этом оставалась в состоянии «обучается» навсегда:
    страница бесконечно ждала новых эпох, а удалить такой эксперимент было
    нельзя. При запуске системы фоновых потоков ещё нет, поэтому все
    незавершённые эксперименты заведомо прерваны — они отмечаются как
    несостоявшиеся. Возвращает их число.
    """
    db = get_session()
    try:
        interrupted = db.scalars(select(Experiment).where(Experiment.status.in_(("new", "running")))).all()
        for experiment in interrupted:
            experiment.status, experiment.message = "failed", INTERRUPTED_MESSAGE
            experiment.finished_at = datetime.now()
            db.add(AuditLog(user_id=experiment.created_by, action="train_interrupted",
                            details=f"Эксперимент №{experiment.id} «{experiment.name}»: обучение прервано "
                                    f"остановкой системы"))
        db.commit()
        if interrupted:
            logger.warning("Прерванных обучений закрыто: %s", len(interrupted))
        return len(interrupted)
    finally:
        db.close()


def register_model_file(db, path, name: str = None, user_id: int = None, activate: bool = False):
    """
    Создаёт запись об эксперименте для готового файла модели (*.pt).
    Метрики и гиперпараметры читаются из самого файла.
    """
    loaded = registry.load_model(path)
    meta = loaded.meta
    experiment = Experiment(
        name=name or f"Модель из файла {path.name}", model_type=loaded.kind,
        params_json=json.dumps(meta.get("train_config") or meta.get("model_config"), ensure_ascii=False),
        metrics_json=json.dumps(meta.get("metrics", {}), ensure_ascii=False),
        status="done", model_file=store.relative(path), created_by=user_id,
        message="Модель загружена из файла", finished_at=datetime.now())
    db.add(experiment)
    db.commit()
    if activate:
        store.activate(db, experiment)
    return experiment


def _register_shipped_models(db) -> None:
    """Создаёт записи об экспериментах для моделей, поставляемых вместе с системой."""
    dataset = store.active_dataset(db)
    for path in sorted(config.MODELS_DIR.glob("model*.pt")):
        try:
            # Первая найденная модель каждого вида становится рабочей
            kind = registry.load_model(path).kind
            title = "Ансамбль многозадачных сетей (итоговая модель)" if kind == "mlp" \
                else "Автокодировщик профилей организаций"
            experiment = register_model_file(db, path, name=title,
                                             activate=store.active_experiment(db, kind) is None)
            experiment.dataset_id = dataset.id if dataset else None
            for record in registry.load_model(path).meta.get("history", []):
                db.add(ExperimentEpoch(experiment_id=experiment.id, epoch=record["epoch"],
                                       train_loss=record.get("train_loss"), val_loss=record.get("val_loss"),
                                       train_metric=record.get("train_metric"),
                                       val_metric=record.get("val_metric"),
                                       learning_rate=record.get("learning_rate")))
            db.commit()
        except Exception as error:   # повреждённый файл не должен мешать запуску
            logger.warning("Файл модели %s пропущен: %s", path.name, error)

    # Модели сравнения (softmax-регрессия, случайный лес) описаны в metrics.json
    summary_path = config.MODELS_DIR / "metrics.json"
    if not summary_path.exists():
        return
    reports = json.loads(summary_path.read_text(encoding="utf-8")).get("reports", {})
    for filename, item in reports.items():
        if not (config.MODELS_DIR / filename).exists():
            continue
        experiment = Experiment(
            name=item["name"], model_type=item["type"], status="done", message="Модель поставляется с системой",
            params_json=json.dumps(item["params"], ensure_ascii=False),
            metrics_json=json.dumps(item["metrics"], ensure_ascii=False),
            model_file=store.relative(config.MODELS_DIR / filename), finished_at=datetime.now(),
            dataset_id=dataset.id if dataset else None)
        db.add(experiment)
        db.flush()
        for record in item.get("history", []):
            db.add(ExperimentEpoch(experiment_id=experiment.id, epoch=record["epoch"],
                                   train_loss=record.get("train_loss"), val_loss=record.get("val_loss")))
    db.commit()


def ensure_default_dataset() -> None:
    """Регистрирует поставляемый набор данных и модели, если база ещё не наполнена."""
    db = get_session()
    try:
        if db.scalar(select(func.count()).select_from(Dataset)) == 0 \
                and store.dataset_is_complete(store.relative(config.MONITORING_DIR)):
            location = store.relative(config.MONITORING_DIR)
            frames = store.frames(location)
            years = sorted(int(y) for y in frames["institutions"]["year"].unique())
            dataset = Dataset(name="Мониторинг вузов — данные с сайта ГИВЦ Минобрнауки", source="site",
                              location=location, years=", ".join(map(str, years)),
                              rows_count=len(frames["institutions"]),
                              description="Информационно-аналитические материалы мониторинга эффективности "
                                          "деятельности образовательных организаций высшего образования")
            db.add(dataset)
            db.commit()
            # Поставляемый набор явно назначается рабочим: наборы, добавленные позже, его не подменят
            store.set_active_dataset(db, dataset.id)
            if db.scalar(select(func.count()).select_from(University)) == 0:
                logger.info("Первый запуск: загрузка набора данных в базу")
                load_monitoring_frames(db, frames)

        if db.scalar(select(func.count()).select_from(Experiment)) == 0:
            _register_shipped_models(db)

        # «Своя» организация для страницы обзора: университет имени С.Ю. Витте
        home = db.get(Setting, "home_university_id")
        if home is not None and not home.value:
            own = db.scalars(select(University).where(University.name.like("%Витте%"),
                                                      University.is_branch.is_(False))).first()
            if own is not None:
                home.value = str(own.id)
                db.commit()

        # Первый прогноз строится автоматически, чтобы разделы не были пустыми
        if db.scalar(select(func.count()).select_from(Prediction)) == 0 \
                and store.active_experiment(db, "mlp") is not None and store.active_dataset(db) is not None:
            from webapp.services import jobs
            try:
                jobs.build_forecast(db)
            except RuntimeError as error:
                logger.warning("Прогноз при первом запуске не построен: %s", error)
    finally:
        db.close()
