# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль session — подключение к базе данных и её первичное наполнение.

Здесь создаётся «движок» SQLAlchemy, фабрика сессий и выполняется
инициализация: создание таблиц, ролей, учётных записей по умолчанию
и настроек системы.
"""

from sqlalchemy import create_engine, event, select
from sqlalchemy.orm import scoped_session, sessionmaker
from werkzeug.security import generate_password_hash

from vuzmon import config
from vuzmon.db.models import Base, Role, Setting, User

_engine = None
_session_factory = None

# Роли пользователей: код, название, описание
DEFAULT_ROLES = [
    ("admin", "Администратор", "Управление пользователями, настройками и данными, полный доступ"),
    ("analyst", "Аналитик", "Загрузка данных, обучение моделей, построение прогнозов и отчётов"),
    ("viewer", "Руководитель", "Просмотр показателей, прогнозов и выгрузка готовых отчётов"),
]

# Учётные записи для первого входа (тестовые; пароли меняются в личном кабинете)
DEFAULT_USERS = [
    ("admin", "Admin#2026", "Администратор системы", "admin", "Системный администратор"),
    ("analyst", "Analyst#2026", "Аналитик центра статистики", "analyst", "Ведущий аналитик"),
    ("viewer", "Viewer#2026", "Руководитель подразделения", "viewer", "Проректор"),
]

# Настройки системы: ключ, значение по умолчанию, пояснение для панели настроек
DEFAULT_SETTINGS = [
    ("organization_name", "Московский университет имени С.Ю. Витте", "Название организации в отчётах"),
    ("home_university_id", "", "Код «своей» организации на сайте мониторинга (для сравнения)"),
    ("base_year", "2025", "Последний год мониторинга, по которому есть данные"),
    ("records_per_page", "25", "Число строк на странице в списках"),
    ("risk_threshold", "0.5", "Порог вероятности для включения организации в зону риска"),
    ("anomaly_quantile", "0.95", "Квантиль ошибки восстановления, выше которого запись считается нетипичной"),
    ("collector_interval", "0.25", "Пауза между запросами к сайту мониторинга, с"),
]


def hash_password(password: str) -> str:
    """Хеш пароля (PBKDF2-SHA256 с солью): сам пароль в базе не хранится."""
    return generate_password_hash(password, method="pbkdf2:sha256")


def get_engine(url: str = None):
    """Возвращает общий «движок» подключения к базе данных (создаёт при первом вызове)."""
    global _engine
    if _engine is None or url is not None:
        config.ensure_dirs()
        _engine = create_engine(url or config.DATABASE_URL, future=True)
        if _engine.dialect.name == "sqlite":
            # В SQLite контроль внешних ключей включается отдельно для каждого соединения
            @event.listens_for(_engine, "connect")
            def _enable_foreign_keys(dbapi_connection, _record):
                cursor = dbapi_connection.cursor()
                cursor.execute("PRAGMA foreign_keys=ON")
                cursor.close()
    return _engine


def get_session_factory(url: str = None):
    """Фабрика сессий, привязанных к текущему потоку (удобно для веб-приложения)."""
    global _session_factory
    if _session_factory is None or url is not None:
        _session_factory = scoped_session(sessionmaker(bind=get_engine(url), expire_on_commit=False))
    return _session_factory


def get_session():
    """Сессия базы данных для текущего потока."""
    return get_session_factory()()


def init_db(url: str = None) -> None:
    """Создаёт таблицы и добавляет роли, пользователей и настройки по умолчанию."""
    engine = get_engine(url)
    Base.metadata.create_all(engine)
    session = get_session_factory(url)()
    try:
        roles = {role.code: role for role in session.scalars(select(Role))}
        for code, title, description in DEFAULT_ROLES:
            if code not in roles:
                roles[code] = Role(code=code, title=title, description=description)
                session.add(roles[code])
        session.flush()

        existing_logins = set(session.scalars(select(User.login)))
        for login, password, full_name, role_code, position in DEFAULT_USERS:
            if login not in existing_logins:
                session.add(User(login=login, password_hash=hash_password(password), full_name=full_name,
                                 role_id=roles[role_code].id, position=position))

        existing_keys = set(session.scalars(select(Setting.key)))
        for key, value, title in DEFAULT_SETTINGS:
            if key not in existing_keys:
                session.add(Setting(key=key, value=value, title=title))
        session.commit()
    finally:
        session.close()


def get_setting(session, key: str, default: str = "") -> str:
    """Значение настройки по ключу (или значение по умолчанию)."""
    setting = session.get(Setting, key)
    return setting.value if setting is not None and setting.value != "" else default
