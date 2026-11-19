# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Интеграционные тесты веб-приложения: вход, разграничение доступа, основные
сценарии (обучение модели, прогноз, выгрузка документов, администрирование).
"""

import io
import json
import zipfile
from datetime import timedelta

import pytest
from docx import Document
from openpyxl import load_workbook
from sqlalchemy import select

from tests.conftest import login
from vuzmon import config
from vuzmon.collect.crawler import DATASET_FILES
from vuzmon.db.models import (AuditLog, Dataset, Experiment, FeedbackMessage, Prediction, Setting, University,
                              User)
from vuzmon.db.session import get_number_setting, get_session, init_db
from vuzmon.ml import pipeline
from webapp.forms import to_int
from webapp.services import jobs, store
from webapp.views_auth import LOCK_MINUTES, MAX_FAILED_LOGINS

PUBLIC_PAGES = ["/", "/universities", "/compare", "/forecast", "/profiles", "/reports/", "/profile", "/help",
                "/feedback/"]
ANALYST_PAGES = ["/data/", "/models/", "/models/new"]
ADMIN_PAGES = ["/admin/users", "/admin/users/new", "/admin/feedback", "/admin/audit", "/admin/settings",
               "/admin/system", "/admin/roles"]


def test_anonymous_user_is_redirected_to_login(client):
    for page in PUBLIC_PAGES + ANALYST_PAGES + ADMIN_PAGES:
        response = client.get(page)
        assert response.status_code == 302 and "/login" in response.headers["Location"], page


def test_login_rejects_wrong_password_and_logs_attempt(client):
    response = client.post("/login", data={"login": "admin", "password": "wrong-password"})
    assert response.status_code == 200 and "Неверный логин или пароль" in response.get_data(as_text=True)
    assert client.get("/").status_code == 302
    last = get_session().scalars(select(AuditLog).order_by(AuditLog.id.desc())).first()
    assert last.action == "login_failed"


@pytest.mark.parametrize("target, location", [
    ("//evil.example/", "/"),
    ("https://evil.example/", "/"),
    ("/\t/evil.example", "/"),          # ошибка: браузер отбрасывает табуляцию и переходит на //evil.example
    ("/\n/evil.example", "/"),
    ("/\\evil.example", "/"),          # обратную косую черту браузеры читают как обычную
    ("forecast", "/"),
    ("/forecast?predicted=0", "/forecast?predicted=0"),      # внутренний адрес сохраняется
])
def test_login_does_not_follow_external_redirect(client, target, location):
    response = client.post("/login", query_string={"next": target}, data={"login": "admin", "password": "Admin#2026"})
    assert response.status_code == 302 and response.headers["Location"] == location
    client.post("/logout")


def test_login_is_suspended_after_repeated_failures(client):
    # Недочёт: число попыток входа не ограничивалось, пароль можно было подбирать перебором
    admin = client.application.test_client()
    login(admin)
    form = {"login": "locktest", "password": "LockTest#2026", "role_id": "3", "is_active": "1"}
    assert admin.post("/admin/users/new", data=form).status_code == 302

    def attempt(password: str, name: str = "locktest"):
        response = client.post("/login", data={"login": name, "password": password})
        return response.status_code, response.get_data(as_text=True)

    # Верный пароль обнуляет счётчик: четыре ошибки, вход, ещё четыре ошибки — вход не приостановлен
    for _ in range(MAX_FAILED_LOGINS - 1):
        assert attempt("wrong")[0] == 200
    assert attempt("LockTest#2026")[0] == 302
    client.post("/logout")
    for number in range(1, MAX_FAILED_LOGINS):
        code, text = attempt("wrong")
        assert code == 200 and "Неверный логин или пароль" in text
        assert ("Осталось попыток" in text) == (MAX_FAILED_LOGINS - number <= 2)
    assert attempt("wrong")[0] == 200                                  # пятая ошибка подряд
    code, text = attempt("LockTest#2026")                              # теперь не помогает и верный пароль
    assert code == 429 and "Слишком много неудачных попыток входа" in text and "мин." in text
    assert client.get("/").status_code == 302
    db = get_session()
    assert db.scalars(select(AuditLog).order_by(AuditLog.id.desc())).first().action == "login_blocked"

    # Другие учётные записи продолжают работать; несуществующий логин приостанавливается так же
    assert login(client.application.test_client(), "analyst", "Analyst#2026").status_code == 302
    for _ in range(MAX_FAILED_LOGINS):
        assert attempt("wrong", "nobody")[0] == 200
    assert attempt("wrong", "nobody")[0] == 429

    # Через LOCK_MINUTES минут вход снова возможен (время имитируется сдвигом записей журнала)
    for row in db.scalars(select(AuditLog).where(AuditLog.action == "login_failed",
                                                 AuditLog.details.like("%locktest"))):
        row.created_at -= timedelta(minutes=LOCK_MINUTES + 1)
    db.commit()
    assert attempt("LockTest#2026")[0] == 302


@pytest.mark.parametrize("username, password, allowed, forbidden", [
    ("admin", "Admin#2026", PUBLIC_PAGES + ANALYST_PAGES + ADMIN_PAGES, []),
    ("analyst", "Analyst#2026", PUBLIC_PAGES + ANALYST_PAGES, ADMIN_PAGES),
    ("viewer", "Viewer#2026", PUBLIC_PAGES, ANALYST_PAGES + ADMIN_PAGES),
])
def test_role_based_access(client, username, password, allowed, forbidden):
    assert login(client, username, password).status_code == 302
    for page in allowed:
        assert client.get(page).status_code == 200, f"{username}: {page}"
    for page in forbidden:
        assert client.get(page).status_code == 403, f"{username}: {page}"


def test_viewer_cannot_change_data(client):
    login(client, "viewer", "Viewer#2026")
    assert client.post("/forecast/build").status_code == 403
    assert client.post("/data/collect", data={"years": "2025"}).status_code == 403
    university = get_session().scalars(select(University)).first()
    assert client.post(f"/universities/{university.id}", data={"text": "заметка"}).status_code == 403


def test_csrf_protection_is_enforced_outside_tests(app):
    app.config["TESTING"] = False
    try:
        fresh = app.test_client()
        assert fresh.post("/login", data={"login": "admin", "password": "Admin#2026"}).status_code == 400
        page = fresh.get("/login").get_data(as_text=True)
        token = page.split('name="_csrf" value="')[1].split('"')[0]
        response = fresh.post("/login", data={"login": "admin", "password": "Admin#2026", "_csrf": token})
        assert response.status_code == 302
    finally:
        app.config["TESTING"] = True


def test_training_forecast_and_exports(client):
    """Сквозной сценарий аналитика: обучить модели, построить прогноз, выгрузить документы."""
    login(client, "analyst", "Analyst#2026")
    db = get_session()
    for model_type, extra in (("mlp", {"hidden": "32", "epochs": "12", "patience": "0"}),
                              ("autoencoder", {"hidden": "16", "epochs": "10", "latent_dim": "3", "n_clusters": "3"})):
        before = {e.id for e in db.scalars(select(Experiment))}
        form = {"model_type": model_type, "name": f"тест {model_type}", "batch_norm": "1", **extra}
        # Фоновый запуск подменяем синхронным, чтобы тест дождался результата
        original, jobs.start_training = jobs.start_training, lambda experiment_id: jobs._training_job(lambda *a: None, experiment_id)
        try:
            response = client.post("/models/new", data=form)
        finally:
            jobs.start_training = original
        assert response.status_code == 302
        db.expire_all()
        experiment = next(e for e in db.scalars(select(Experiment)) if e.id not in before)
        assert experiment.status == "done", experiment.message
        assert experiment.is_active and len(experiment.epochs) >= 10
        page = client.get(f"/models/{experiment.id}")
        assert page.status_code == 200 and "Кривые обучения" in page.get_data(as_text=True)
        assert client.get(f"/models/{experiment.id}/download").status_code == 200
        state = client.get(f"/models/{experiment.id}/status.json").get_json()
        assert state["status"] == "done"

    classifier = db.scalars(select(Experiment).where(Experiment.model_type == "mlp")).first()
    metrics = json.loads(classifier.metrics_json)
    assert 0.4 < metrics["test"]["accuracy"] <= 1.0 and len(metrics["importance"]) > 0

    assert client.post("/forecast/build").status_code == 302
    db.expire_all()
    predictions = db.scalars(select(Prediction)).all()
    assert predictions and all(p.anomaly_score is not None for p in predictions)
    university_id = predictions[0].university_id
    assert "Что повлияло на прогноз" in client.get(f"/universities/{university_id}").get_data(as_text=True)
    assert client.get("/profiles").status_code == 200
    scenario = client.post(f"/forecast/what-if/{university_id}",
                           data={"e1": "95", "e2": "900", "e3": "30", "e4": "20000", "e8": "9"})
    assert scenario.status_code == 200 and "По сценарию" in scenario.get_data(as_text=True)

    for url, signature in ((f"/reports/university/{university_id}.docx", b"PK"), ("/reports/forecast.xlsx", b"PK"),
                           ("/reports/summary.docx", b"PK"), (f"/reports/experiment/{classifier.id}.docx", b"PK")):
        response = client.get(url)
        assert response.status_code == 200 and response.data[:2] == signature, url

    # Файл модели можно загрузить обратно как новую модель
    model_bytes = client.get(f"/models/{classifier.id}/download").data
    response = client.post("/models/upload", data={"file": (io.BytesIO(model_bytes), "saved.pt"), "name": "из файла"},
                           content_type="multipart/form-data")
    assert response.status_code == 302
    assert "готова к применению" in client.get(response.headers["Location"]).get_data(as_text=True)
    bad = client.post("/models/upload", data={"file": (io.BytesIO(b"not a model"), "bad.pt")},
                      content_type="multipart/form-data", follow_redirects=True)
    assert "не удалось загрузить" in bad.get_data(as_text=True)


def test_dataset_download_and_upload_roundtrip(client):
    login(client, "analyst", "Analyst#2026")
    archive = client.get("/data/1/download")
    assert archive.status_code == 200
    names = zipfile.ZipFile(io.BytesIO(archive.data)).namelist()
    assert "institutions.csv" in names and len(names) == 5
    response = client.post("/data/upload", data={"files": (io.BytesIO(archive.data), "dataset.zip"), "name": "копия"},
                           content_type="multipart/form-data")
    assert response.status_code == 302 and "/data/2" in response.headers["Location"]
    assert client.get("/data/2").status_code == 200
    broken = client.post("/data/upload", data={"files": (io.BytesIO(b"a,b\n1,2\n"), "institutions.csv")},
                         content_type="multipart/form-data", follow_redirects=True)
    assert "не загружен" in broken.get_data(as_text=True)
    assert client.post("/data/2/delete").status_code == 302 and client.get("/data/2").status_code == 404


def _archive(frames: dict) -> io.BytesIO:
    """ZIP-архив с пятью CSV-файлами набора данных."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for table, filename in DATASET_FILES.items():
            archive.writestr(filename, frames[table].to_csv(index=False))
    buffer.seek(0)
    return buffer


def _upload_dataset(client, frames: dict, name: str) -> int:
    """Загружает набор данных ZIP-архивом и возвращает его номер."""
    response = client.post("/data/upload", data={"files": (_archive(frames), "dataset.zip"), "name": name},
                           content_type="multipart/form-data")
    assert response.status_code == 302
    return int(response.headers["Location"].rsplit("/", 1)[1])


def test_new_dataset_does_not_replace_working_one(client, small_frames):
    # Ошибка, найденная при проверке установки: пока рабочий набор не задан настройкой (так было
    # в поставляемой базе), рабочим считался набор, созданный последним, — в том числе пробный
    login(client, "analyst", "Analyst#2026")
    db = get_session()
    working_id = store.active_dataset(db).id
    db.delete(db.get(Setting, "active_dataset_id"))
    db.commit()
    try:
        new_id = _upload_dataset(client, small_frames, "ещё один набор")
        assert new_id != working_id
        assert store.active_dataset(get_session()).id == working_id
        assert client.get("/").status_code == 200
        assert client.post(f"/data/{new_id}/delete").status_code == 302
    finally:
        store.set_active_dataset(get_session(), working_id)


def test_short_dataset_cannot_become_working(client, small_frames):
    # Набор за один год нельзя разбить по времени, поэтому рабочим он не становится
    login(client, "analyst", "Analyst#2026")
    last_year = int(small_frames["institutions"]["year"].max())
    one_year = dict(small_frames)
    for table, column in (("institutions", "year"), ("medians", "year"), ("extra", "year"), ("main", "page_year")):
        one_year[table] = small_frames[table][small_frames[table][column] == last_year]
    new_id = _upload_dataset(client, one_year, "один год")
    db = get_session()
    working_id = store.active_dataset(db).id
    universities = len(db.scalars(select(University)).all())

    page = client.post(f"/data/{new_id}/activate", follow_redirects=True).get_data(as_text=True)
    assert "нельзя сделать рабочим" in page and "за четыре года" in page
    db = get_session()
    assert store.active_dataset(db).id == working_id                        # рабочий набор прежний
    assert len(db.scalars(select(University)).all()) == universities        # данные в базе не заменены
    assert client.get("/").status_code == 200 and client.get("/forecast").status_code == 200
    assert client.post(f"/data/{new_id}/delete").status_code == 302


def test_upload_rejects_dataset_with_wrong_values(client, small_frames, monkeypatch):
    # Ошибка: текст вместо года давал ошибку 500 при загрузке, текст вместо показателя — при выборе набора рабочим
    login(client, "analyst", "Analyst#2026")
    datasets_dir = config.DATA_DIR / "datasets"
    folders = set(datasets_dir.iterdir()) if datasets_dir.exists() else set()
    count = len(get_session().scalars(select(Dataset)).all())

    def upload(frames: dict) -> str:
        page = client.post("/data/upload", data={"files": (_archive(frames), "dataset.zip"), "name": "с ошибкой"},
                           content_type="multipart/form-data", follow_redirects=True)
        assert page.status_code == 200
        return page.get_data(as_text=True)

    wrong_year = dict(small_frames, institutions=small_frames["institutions"].astype({"year": object}))
    wrong_year["institutions"].iloc[0, wrong_year["institutions"].columns.get_loc("year")] = "две тысячи"
    text = upload(wrong_year)
    assert "Набор данных не загружен" in text and "institutions.csv" in text
    assert "ожидается целое число" in text and "в строке 2 указано «две тысячи»" in text

    wrong_value = dict(small_frames, main=small_frames["main"].astype({"e1": object}))
    wrong_value["main"].iloc[3, wrong_value["main"].columns.get_loc("e1")] = "много"
    text = upload(wrong_value)
    assert "main_indicators.csv" in text and "ожидается число" in text and "в строке 5 указано «много»" in text

    text = upload({name: frame.iloc[0:0] for name, frame in small_frames.items()})
    assert "нет ни одной записи" in text

    # Наборы с ошибками не попадают ни в перечень, ни на диск
    assert len(get_session().scalars(select(Dataset)).all()) == count
    assert (set(datasets_dir.iterdir()) if datasets_dir.exists() else set()) == folders

    # Два набора, добавленные в одну секунду, получают разные каталоги
    first, second = store.new_dataset_directory("check"), store.new_dataset_directory("check")
    assert first != second and first.is_dir() and second.is_dir()
    first.rmdir()
    second.rmdir()

    # Набор, подготовка которого завершается любым сбоем, рабочим не становится — страницы продолжают открываться
    new_id = _upload_dataset(client, small_frames, "исправный набор")
    working_id = store.active_dataset(get_session()).id

    def broken(location):
        raise TypeError("unsupported operand type(s) for /: 'str' and 'float'")
    monkeypatch.setattr(store, "prepared", broken)
    page = client.post(f"/data/{new_id}/activate", follow_redirects=True).get_data(as_text=True)
    assert "нельзя сделать рабочим" in page and "не удалось обработать" in page
    monkeypatch.undo()
    assert store.active_dataset(get_session()).id == working_id and client.get("/").status_code == 200
    assert client.post(f"/data/{new_id}/delete").status_code == 302


def test_unreadable_files_and_failed_training_are_reported_in_russian(client, monkeypatch):
    # Недочёт: при нечитаемом файле пользователь видел служебное сообщение библиотеки на английском языке
    login(client, "analyst", "Analyst#2026")

    def upload(files: dict, url: str = "/data/upload") -> str:
        page = client.post(url, data=files, content_type="multipart/form-data", follow_redirects=True)
        assert page.status_code == 200
        return page.get_data(as_text=True)

    def archive(content: bytes) -> io.BytesIO:
        buffer = io.BytesIO()
        with zipfile.ZipFile(buffer, "w") as packed:
            for filename in DATASET_FILES.values():
                packed.writestr(filename, content)
        buffer.seek(0)
        return buffer

    text = upload({"files": (io.BytesIO(b"not a zip"), "data.zip")})
    assert "файл data.zip не является ZIP-архивом или повреждён" in text and "not a zip file" not in text
    text = upload({"files": (archive(b"\xff\xfe\x00bad"), "data.zip")})
    assert "должен быть в кодировке UTF-8" in text and "codec" not in text
    text = upload({"files": (archive(b""), "data.zip")})
    assert "файл institutions.csv пуст" in text and "No columns" not in text

    text = upload({"file": (io.BytesIO(b"hello world"), "fake.pt")}, "/models/upload")
    assert "Файл не удалось загрузить как модель" in text and "повреждён или создан другой программой" in text

    # Сбой обучения: на странице эксперимента — краткая причина без трассировки и путей к файлам сервера
    def failing(*args, **kwargs):
        raise RuntimeError("size mismatch")
    monkeypatch.setattr(pipeline, "run_mlp", failing)
    db = get_session()
    experiment = Experiment(name="сбой обучения", model_type="mlp", params_json="{}", status="new",
                            dataset_id=store.active_dataset(db).id)
    db.add(experiment)
    db.commit()
    with pytest.raises(RuntimeError):
        jobs._training_job(lambda *a: None, experiment.id)
    db.expire_all()
    experiment = db.get(Experiment, experiment.id)
    assert experiment.status == "failed"
    assert experiment.message == "Обучение не выполнено. Причина: RuntimeError: size mismatch"
    page = client.get(f"/models/{experiment.id}").get_data(as_text=True)
    assert "Обучение завершилось ошибкой" in page and "Traceback" not in page
    assert client.post(f"/models/{experiment.id}/delete").status_code == 302


def test_server_error_page_is_in_russian(app, client, monkeypatch):
    # Необработанная ошибка показывается страницей системы, а не служебной страницей веб-сервера
    def broken_page():
        raise RuntimeError("сбой при формировании страницы")
    monkeypatch.setitem(app.config, "PROPAGATE_EXCEPTIONS", False)
    monkeypatch.setitem(app.view_functions, "auth.help_page", broken_page)
    login(client)
    response = client.get("/help")
    assert response.status_code == 500
    assert "Внутренняя ошибка" in response.get_data(as_text=True)


def test_notes_and_search(client):
    login(client, "analyst", "Analyst#2026")
    university = get_session().scalars(select(University).where(University.is_branch.is_(False))).first()
    client.post(f"/universities/{university.id}", data={"text": "Проверить данные о доходах"})
    page = client.get(f"/universities/{university.id}").get_data(as_text=True)
    assert "Проверить данные о доходах" in page
    word = university.short_name.split()[0]
    assert university.short_name in client.get(f"/universities?q={word}").get_data(as_text=True)
    assert "ничего не найдено" in client.get("/universities?q=несуществующийвуз").get_data(as_text=True)


def test_compare_with_repeated_and_same_named_organizations(client):
    # Ошибка: повторный выбор организации и две организации с одинаковым названием давали ошибку 500
    login(client, "viewer", "Viewer#2026")
    db = get_session()
    first, second = db.scalars(select(University).where(University.is_branch.is_(False)).limit(2)).all()
    page = client.get(f"/compare?id={first.id}&id={first.id}")
    assert page.status_code == 200 and page.get_data(as_text=True).count(">убрать<") == 1
    original = second.short_name
    second.short_name = first.short_name
    db.commit()
    try:
        page = client.get(f"/compare?id={first.id}&id={second.id}")
        assert page.status_code == 200 and page.get_data(as_text=True).count(">убрать<") == 2
    finally:
        second.short_name = original
        db.commit()


def test_malformed_numbers_in_forms_do_not_break_pages(client):
    # Ошибка: нечисловое значение в числовом поле формы или адреса страницы давало ошибку 500
    assert to_int("7") == 7 and to_int("семь", 3) == 3 and to_int(None, 1) == 1 and to_int("500", 0, 0, 100) == 100
    login(client)
    form = {"login": "tester1", "password": "Tester#2026", "role_id": "abc", "is_active": "1"}
    page = client.post("/admin/users/new", data=form)
    assert page.status_code == 200 and "Выберите роль" in page.get_data(as_text=True)
    assert get_session().scalars(select(User).where(User.login == "tester1")).first() is None
    assert client.get("/admin/audit?days=99999999999").status_code == 200
    assert client.get("/admin/audit?days=abc").status_code == 200
    university = get_session().scalars(select(University)).first()
    removal = client.post(f"/universities/{university.id}", data={"action": "delete_note", "note_id": "abc"})
    assert removal.status_code == 302


def test_admin_manages_users_and_settings(client):
    login(client)
    form = {"login": "expert", "password": "Expert#2026", "full_name": "Эксперт", "role_id": "3", "is_active": "1"}
    assert client.post("/admin/users/new", data=form).status_code == 302
    weak = client.post("/admin/users/new", data={**form, "login": "weak", "password": "123"})
    assert "не менее 8 символов" in weak.get_data(as_text=True)
    duplicate = client.post("/admin/users/new", data=form)
    assert "уже существует" in duplicate.get_data(as_text=True)

    db = get_session()
    created = db.scalars(select(User).where(User.login == "expert")).one()
    assert created.password_hash != "Expert#2026" and created.role.code == "viewer"
    other = client.application.test_client()
    assert login(other, "expert", "Expert#2026").status_code == 302 and other.get("/").status_code == 200
    # Отключённая учётная запись теряет доступ сразу, даже при действующей сессии
    client.post(f"/admin/users/{created.id}", data={**form, "password": "", "is_active": ""})
    assert other.get("/").status_code == 302

    me = db.scalars(select(User).where(User.login == "admin")).one()
    locked = client.post(f"/admin/users/{me.id}", data={"role_id": "3", "is_active": "1", "password": ""})
    assert "Нельзя отключить собственную" in locked.get_data(as_text=True)

    client.post("/admin/settings", data={"records_per_page": "10", "organization_name": "Тест"})
    assert client.get("/universities").get_data(as_text=True).count("/universities/") <= 12
    assert client.get("/reports/audit.xlsx").data[:2] == b"PK"
    assert "settings_update" in client.get("/admin/audit").get_data(as_text=True)


def test_settings_are_validated(client):
    # Ошибка: нечисловое число строк на странице сохранялось, после чего раздел «Прогноз» возвращал ошибку 500
    login(client)
    db = get_session()
    before = {row.key: row.value for row in db.scalars(select(Setting))}
    page = client.post("/admin/settings", data={"records_per_page": "abc", "collector_interval": "быстро",
                                                "risk_threshold": "5", "home_university_id": "999999999"})
    text = page.get_data(as_text=True)
    assert page.status_code == 200 and "Настройки не сохранены" in text
    assert "введите целое число от 5 до 200" in text and "значение должно быть от 0,05 до 0,95" in text
    assert "организации с таким кодом нет" in text and 'value="abc"' in text       # введённое показано снова
    db.expire_all()
    assert {row.key: row.value for row in db.scalars(select(Setting))} == before   # не сохранено ничего

    # Десятичная запятая допустима; значение хранится в единой записи
    client.post("/admin/settings", data={"collector_interval": "0,5"})
    db.expire_all()
    assert db.get(Setting, "collector_interval").value == "0.5"
    client.post("/admin/settings", data={"collector_interval": before["collector_interval"]})

    # Код рабочего набора данных меняется только в разделе «Данные», в настройках его нет
    assert "active_dataset_id" in before and "active_dataset_id" not in client.get("/admin/settings").get_data(as_text=True)
    client.post("/admin/settings", data={"active_dataset_id": "999"})
    db.expire_all()
    assert db.get(Setting, "active_dataset_id").value == before["active_dataset_id"]

    # Значение, попавшее в базу в обход панели, не мешает открывать страницы
    db.get(Setting, "records_per_page").value = "abc"
    db.commit()
    try:
        assert client.get("/forecast").status_code == 200 and client.get("/universities").status_code == 200
        assert get_number_setting(db, "records_per_page", 25, 5, 200) == 25
    finally:
        db.get(Setting, "records_per_page").value = before["records_per_page"]
        db.commit()


def test_settings_take_effect(client):
    # Недочёт: название организации, порог зоны риска и квантиль нетипичности в настройках ни на что не влияли
    login(client)
    db = get_session()
    # Настройка «последний год мониторинга» ничем не управляла: при запуске системы она удаляется из базы
    db.add(Setting(key="base_year", value="2025", title="Последний год мониторинга"))
    db.commit()
    init_db()
    db = get_session()
    before = {row.key: row.value for row in db.scalars(select(Setting))}
    assert "base_year" not in before and "base_year" not in client.get("/admin/settings").get_data(as_text=True)
    risks = [json.loads(p.probabilities_json)[0] for p in db.scalars(select(Prediction))]
    assert risks                                           # прогноз построен в сквозном сценарии выше

    def rows_in_excel(query: str) -> int:
        return load_workbook(io.BytesIO(client.get(f"/reports/forecast.xlsx{query}").data)).active.max_row

    try:
        counts = []
        for threshold in ("0.05", "0.6"):
            client.post("/admin/settings", data={"risk_threshold": threshold})
            expected = sum(1 for value in risks if value >= float(threshold))
            page = client.get("/forecast?risk=1").get_data(as_text=True)
            assert f"у <b>{expected}</b> организаций" in page and f"Организаций в выборке: <b>{expected}</b>" in page
            assert rows_in_excel("") - rows_in_excel("?risk=1") == len(risks) - expected    # в файле тот же отбор
            counts.append(expected)
        assert counts[0] > counts[1]

        client.post("/admin/settings", data={"organization_name": "Проверочный университет", "anomaly_quantile": "0.9"})
        summary = Document(io.BytesIO(client.get("/reports/summary.docx").data))
        text = " ".join(paragraph.text for paragraph in summary.paragraphs)
        assert "(Проверочный университет)" in text and f"не ниже 60,0 % — у {counts[1]} организаций" in text
        assert 'name="anomaly_quantile" value="0.9"' in client.get("/models/new").get_data(as_text=True)
    finally:
        client.post("/admin/settings", data={key: before[key] for key in ("risk_threshold", "organization_name",
                                                                         "anomaly_quantile")})


def test_feedback_form_validates_input(client):
    login(client, "analyst", "Analyst#2026")
    before = len(get_session().scalars(select(FeedbackMessage)).all())
    short = client.post("/feedback/", data={"category": "question", "subject": "Тема", "text": "коротко"})
    page = short.get_data(as_text=True)
    assert short.status_code == 200 and "Тема: от 5 до 160 символов" in page and "Текст обращения: от 10" in page
    assert 'value="Тема"' in page and "коротко</textarea>" in page         # введённое не теряется
    unknown = client.post("/feedback/", data={"category": "spam", "subject": "Обычная тема",
                                              "text": "Достаточно длинный текст"})
    assert "Выберите вид обращения" in unknown.get_data(as_text=True)
    assert len(get_session().scalars(select(FeedbackMessage)).all()) == before


def test_feedback_message_and_admin_reply(client):
    """Сквозной сценарий обратной связи: обращение пользователя — ответ администратора."""
    login(client, "analyst", "Analyst#2026")
    form = {"category": "data", "subject": "Расхождение по показателю E.2",
            "text": "В карточке организации значение за 2024 год\nне совпадает с сайтом мониторинга."}
    sent = client.post("/feedback/", data=form, follow_redirects=True)
    assert "Обращение отправлено администратору" in sent.get_data(as_text=True)
    db = get_session()
    message = db.scalars(select(FeedbackMessage).order_by(FeedbackMessage.id.desc())).first()
    assert message.status == "new" and message.author.login == "analyst" and message.reply == ""
    assert db.scalars(select(AuditLog).order_by(AuditLog.id.desc())).first().action == "feedback_send"
    assert "ответа пока нет" in client.get("/feedback/").get_data(as_text=True)

    # Чужие обращения пользователю не видны, а отвечать может только администратор
    other = client.application.test_client()
    login(other, "viewer", "Viewer#2026")
    assert form["subject"] not in other.get("/feedback/").get_data(as_text=True)
    assert other.post(f"/admin/feedback/{message.id}", data={"reply": "ответ", "status": "closed"}).status_code == 403
    assert client.post(f"/admin/feedback/{message.id}", data={"reply": "ответ", "status": "closed"}).status_code == 403

    admin = client.application.test_client()
    login(admin)
    listing = admin.get("/admin/feedback?status=new").get_data(as_text=True)
    new_count = len(db.scalars(select(FeedbackMessage).where(FeedbackMessage.status == "new")).all())
    assert form["subject"] in listing and f"Новое: {new_count}" in listing
    empty = admin.post(f"/admin/feedback/{message.id}", data={"reply": "", "status": "closed"}, follow_redirects=True)
    assert "напишите ответ пользователю" in empty.get_data(as_text=True)
    assert admin.post("/admin/feedback/999999", data={"reply": "ответ", "status": "closed"}).status_code == 404
    answer = "Значение исправлено: набор данных за 2024 год собран заново."
    saved = admin.post(f"/admin/feedback/{message.id}", data={"reply": answer, "status": "closed"},
                       follow_redirects=True)
    assert f"Обращение № {message.id} сохранено" in saved.get_data(as_text=True)
    db.expire_all()
    message = db.get(FeedbackMessage, message.id)
    assert message.status == "closed" and message.reply == answer
    assert message.responder.login == "admin" and message.replied_at is not None

    # Автор видит ответ и новое состояние своего обращения
    page = client.get("/feedback/").get_data(as_text=True)
    assert answer in page and "закрыто" in page


def test_profile_password_change(client):
    login(client, "viewer", "Viewer#2026")
    wrong = client.post("/profile", data={"action": "password", "old_password": "bad", "new_password": "NewPass#2026",
                                          "repeat_password": "NewPass#2026"}, follow_redirects=True)
    assert "Текущий пароль указан неверно" in wrong.get_data(as_text=True)
    ok = client.post("/profile", data={"action": "password", "old_password": "Viewer#2026",
                                       "new_password": "NewPass#2026", "repeat_password": "NewPass#2026"},
                     follow_redirects=True)
    assert "Пароль изменён" in ok.get_data(as_text=True)
    fresh = client.application.test_client()
    assert login(fresh, "viewer", "Viewer#2026").status_code == 200       # старый пароль больше не подходит
    assert login(fresh, "viewer", "NewPass#2026").status_code == 302
