# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль store — доступ веб-приложения к наборам данных и файлам моделей.

Чтение CSV-файлов, построение выборки и загрузка моделей занимают заметное
время, поэтому результаты кэшируются в памяти процесса. Ключ кэша включает
время изменения файлов: при замене набора данных или модели кэш обновляется
автоматически.
"""

import threading
from pathlib import Path

from sqlalchemy import select

from vuzmon import config
from vuzmon.collect.crawler import DATASET_FILES, load_frames
from vuzmon.db.models import Dataset, Experiment, Setting
from vuzmon.ml import pipeline, registry

_lock = threading.Lock()
_cache = {}


def resolve(location: str) -> Path:
    """Абсолютный путь по пути из базы данных (в базе пути хранятся относительно корня проекта)."""
    path = Path(location)
    return path if path.is_absolute() else config.BASE_DIR / path


def relative(path) -> str:
    """Путь относительно корня проекта — в таком виде он не зависит от компьютера."""
    path = Path(path).resolve()
    try:
        return path.relative_to(config.BASE_DIR.resolve()).as_posix()
    except ValueError:
        return str(path)


def _signature(directory: Path) -> tuple:
    """Отпечаток набора данных: время изменения каждого файла."""
    return tuple((name, (directory / name).stat().st_mtime_ns)
                 for name in DATASET_FILES.values() if (directory / name).exists())


def _cached(key, signature, builder):
    """Возвращает значение из кэша или строит его заново, если данные изменились."""
    with _lock:
        entry = _cache.get(key)
        if entry is not None and entry[0] == signature:
            return entry[1]
    value = builder()
    with _lock:
        _cache[key] = (signature, value)
    return value


def dataset_is_complete(location: str) -> bool:
    """Все ли пять CSV-файлов набора данных на месте."""
    directory = resolve(location)
    return all((directory / name).exists() for name in DATASET_FILES.values())


def frames(location: str) -> dict:
    """Таблицы набора данных (кэшируются)."""
    directory = resolve(location)
    return _cached(("frames", str(directory)), _signature(directory), lambda: load_frames(directory))


def prepared(location: str) -> pipeline.PreparedData:
    """Подготовленная выборка для обучения и прогноза (кэшируется)."""
    directory = resolve(location)
    return _cached(("prepared", str(directory)), _signature(directory),
                   lambda: pipeline.prepare_data(frames(location)))


def model(model_file: str) -> registry.LoadedModel:
    """Загруженная модель (кэшируется по пути и времени изменения файла)."""
    path = resolve(model_file)
    return _cached(("model", str(path)), path.stat().st_mtime_ns, lambda: registry.load_model(path))


# --------------------------------------------------------------------------
# Активные набор данных и модели
# --------------------------------------------------------------------------
def active_dataset(db):
    """Набор данных, с которым сейчас работает система (по настройке или самый свежий)."""
    setting = db.get(Setting, "active_dataset_id")
    if setting is not None and setting.value.isdigit():
        dataset = db.get(Dataset, int(setting.value))
        if dataset is not None:
            return dataset
    return db.scalars(select(Dataset).order_by(Dataset.id.desc())).first()


def set_active_dataset(db, dataset_id: int) -> None:
    setting = db.get(Setting, "active_dataset_id")
    if setting is None:
        setting = Setting(key="active_dataset_id", title="Код активного набора данных")
        db.add(setting)
    setting.value = str(dataset_id)
    db.commit()


# Виды экспериментов, дающих модель одного назначения: подбор гиперпараметров
# тоже заканчивается обученной нейронной сетью
TYPE_GROUPS = {"mlp": ("mlp", "search"), "autoencoder": ("autoencoder",)}


def active_experiment(db, model_type: str):
    """Эксперимент, модель которого выбрана для применения (mlp — прогноз, autoencoder — профили)."""
    return db.scalars(select(Experiment).where(Experiment.model_type.in_(TYPE_GROUPS[model_type]),
                                               Experiment.is_active.is_(True),
                                               Experiment.status == "done")).first()


def activate(db, experiment) -> None:
    """Делает модель эксперимента рабочей; прежняя модель того же назначения отключается."""
    group = next(types for types in TYPE_GROUPS.values() if experiment.model_type in types)
    for other in db.scalars(select(Experiment).where(Experiment.model_type.in_(group))):
        other.is_active = False
    experiment.is_active = True
    db.commit()


def active_model(db, model_type: str):
    """Пара (эксперимент, загруженная модель) или (None, None), если активной модели нет."""
    experiment = active_experiment(db, model_type)
    if experiment is None or not experiment.model_file or not resolve(experiment.model_file).exists():
        return None, None
    return experiment, model(experiment.model_file)
