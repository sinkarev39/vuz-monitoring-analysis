# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Общие заготовки автоматических тестов (pytest).

Тесты не обращаются к интернету и не трогают рабочую базу данных: набор
данных для проверок — небольшая выборка организаций из поставляемых CSV,
а база создаётся во временном каталоге.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from vuzmon import config  # noqa: E402
from vuzmon.collect.crawler import DATASET_FILES, load_frames  # noqa: E402


@pytest.fixture(scope="session")
def frames():
    """Полный поставляемый набор данных мониторинга."""
    return load_frames(config.MONITORING_DIR)


@pytest.fixture(scope="session")
def small_frames(frames):
    """Уменьшенный набор: 150 головных организаций и 20 филиалов за все годы."""
    institutions = frames["institutions"]
    heads = institutions.loc[institutions["is_branch"] == 0, "inst_id"].drop_duplicates().head(150)
    branches = institutions.loc[institutions["is_branch"] == 1, "inst_id"].drop_duplicates().head(20)
    ids = set(heads) | set(branches)
    return {name: (frame[frame["inst_id"].isin(ids)].reset_index(drop=True) if "inst_id" in frame else frame)
            for name, frame in frames.items()}


@pytest.fixture(scope="session")
def app(tmp_path_factory, small_frames):
    """Веб-приложение на временной базе данных с небольшим набором данных."""
    from vuzmon.db import session as db_session
    from vuzmon.db.loader import load_monitoring_frames
    from vuzmon.db.models import Dataset
    from webapp import create_app
    from webapp.services import store

    folder = tmp_path_factory.mktemp("vuzmon")
    data_dir = folder / "dataset"
    data_dir.mkdir()
    for name, filename in DATASET_FILES.items():
        small_frames[name].to_csv(data_dir / filename, index=False, encoding="utf-8")

    # Модели, обученные в тестах, складываются во временный каталог
    config.MODELS_DIR = folder / "models"
    application = create_app(database_url=f"sqlite:///{folder / 'test.db'}", testing=True,
                             bootstrap_data=False)
    db = db_session.get_session()
    # Тестовая база создаётся пустой: регистрируем уменьшенный набор и делаем его рабочим
    dataset = Dataset(name="Тестовый набор", source="file", location=str(data_dir), years="2021-2025",
                      rows_count=len(small_frames["institutions"]))
    db.add(dataset)
    db.commit()
    load_monitoring_frames(db, small_frames)
    store.set_active_dataset(db, dataset.id)
    yield application
    db_session.get_session_factory().remove()


@pytest.fixture()
def client(app):
    return app.test_client()


def login(client, username="admin", password="Admin#2026"):
    """Вход в систему под указанной учётной записью."""
    return client.post("/login", data={"login": username, "password": password})


@pytest.fixture()
def sample_table():
    """Небольшая таблица признаков с пропусками и «тяжёлым хвостом»."""
    return pd.DataFrame({"a": [1.0, 2.0, 3.0, None, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0],
                         "big": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 1000, 100000],
                         "flag": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]}, dtype="float64")
