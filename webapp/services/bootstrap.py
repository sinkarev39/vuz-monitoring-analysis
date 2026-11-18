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
from vuzmon.db.models import Dataset, Experiment, University
from vuzmon.db.session import get_session
from vuzmon.ml import registry
from webapp.services import store

logger = logging.getLogger(__name__)


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


def ensure_default_dataset() -> None:
    """Регистрирует поставляемый набор данных и модели, если база ещё не наполнена."""
    db = get_session()
    try:
        if db.scalar(select(func.count()).select_from(Dataset)) == 0 \
                and store.dataset_is_complete(store.relative(config.MONITORING_DIR)):
            location = store.relative(config.MONITORING_DIR)
            frames = store.frames(location)
            years = sorted(int(y) for y in frames["institutions"]["year"].unique())
            db.add(Dataset(name="Мониторинг вузов — данные с сайта ГИВЦ Минобрнауки", source="site",
                           location=location, years=", ".join(map(str, years)),
                           rows_count=len(frames["institutions"]),
                           description="Информационно-аналитические материалы мониторинга эффективности "
                                       "деятельности образовательных организаций высшего образования"))
            db.commit()
            if db.scalar(select(func.count()).select_from(University)) == 0:
                logger.info("Первый запуск: загрузка набора данных в базу")
                load_monitoring_frames(db, frames)

        if db.scalar(select(func.count()).select_from(Experiment)) == 0:
            for path in sorted(config.MODELS_DIR.glob("model*.pt")):
                try:
                    # Первая найденная модель каждого вида становится активной
                    kind = registry.load_model(path).kind
                    has_active = store.active_experiment(db, kind) is not None
                    register_model_file(db, path, activate=not has_active)
                except Exception as error:   # повреждённый файл не должен мешать запуску
                    logger.warning("Файл модели %s пропущен: %s", path.name, error)
    finally:
        db.close()
