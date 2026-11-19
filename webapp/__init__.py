# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Пакет webapp — веб-интерфейс системы (клиент-серверная архитектура).

Сервер написан на Flask: он принимает запросы браузера, обращается к базе
данных и к подсистеме машинного обучения (пакет vuzmon) и возвращает
HTML-страницы. Функция create_app собирает приложение: настройки, обработчики
ошибок, общие данные шаблонов и наборы страниц (blueprints).
"""

import secrets
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, request

from vuzmon import config
from vuzmon.db.session import get_session_factory, init_db
from vuzmon.ml.dataset import CLASS_NAMES
from webapp import security

# Пункты главного меню: (имя обработчика, название, значок, роли с доступом)
MENU = [
    ("main.dashboard", "Обзор", "◉", security.ROLE_ANY),
    ("main.universities", "Организации", "▤", security.ROLE_ANY),
    ("main.compare", "Сравнение", "⇄", security.ROLE_ANY),
    ("forecast.index", "Прогноз", "↗", security.ROLE_ANY),
    ("forecast.profiles", "Нетипичные профили", "◎", security.ROLE_ANY),
    ("data.index", "Данные", "⛁", security.ROLE_ANALYST),
    ("models.index", "Модели", "⚙", security.ROLE_ANALYST),
    ("reports.index", "Отчёты", "▦", security.ROLE_ANY),
    ("admin.users", "Администрирование", "⚿", security.ROLE_ADMIN),
    ("auth.profile", "Личный кабинет", "☺", security.ROLE_ANY),
    ("feedback.index", "Обратная связь", "✉", security.ROLE_ANY),
    ("auth.help_page", "Справка", "?", security.ROLE_ANY),
]


# Политика безопасности содержимого: страницы загружают сценарии, стили и изображения только с самого
# сервера системы (графики — встроенные SVG, поэтому для стилей разрешены атрибуты style), формы
# отправляются только на него же, а показ страниц во фрейме другого сайта запрещён
CONTENT_SECURITY_POLICY = ("default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
                           "object-src 'none'; base-uri 'self'; form-action 'self'; frame-ancestors 'none'")


# Страницы, которых нет в меню, подсвечивают «родительский» пункт
MENU_PARENTS = {"main.university": "main.universities", "forecast.what_if": "forecast.index"}


def active_menu(endpoint: str) -> str:
    """Пункт меню, который нужно подсветить для текущей страницы."""
    endpoints = [item[0] for item in MENU]
    if endpoint in endpoints:
        return endpoint
    if endpoint in MENU_PARENTS:
        return MENU_PARENTS[endpoint]
    blueprint = endpoint.split(".")[0]
    return next((item for item in endpoints if item.split(".")[0] == blueprint), "")


def _secret_key() -> str:
    """Ключ подписи сессий: создаётся при первом запуске и хранится вне репозитория."""
    path = config.BASE_DIR / "instance" / "secret_key"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(secrets.token_hex(32), encoding="utf-8")
    return path.read_text(encoding="utf-8").strip()


def format_number(value, digits: int = 2) -> str:
    """Число в российской записи: пробел между тысячами, запятая перед дробной частью."""
    if value is None or value != value:   # None или NaN
        return "—"
    if isinstance(value, float) and not value.is_integer():
        text = f"{value:,.{digits}f}"
    else:
        text = f"{int(value):,}"
    return text.replace(",", " ").replace(".", ",")


def create_app(database_url: str = None, testing: bool = False, bootstrap_data: bool = True) -> Flask:
    """
    Создаёт и настраивает экземпляр веб-приложения.

    bootstrap_data — выполнять ли первичное наполнение (загрузка поставляемого
    набора данных, регистрация моделей, первый прогноз); в тестах отключается.
    """
    app = Flask(__name__)
    app.config.update(
        SECRET_KEY="test-secret" if testing else _secret_key(),
        TESTING=testing,
        MAX_CONTENT_LENGTH=64 * 1024 * 1024,     # предельный размер загружаемого файла — 64 МБ
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
    )
    init_db(database_url)

    # ------------------------------------------------ общие обработчики
    @app.before_request
    def _protect_forms():
        if not app.config["TESTING"]:
            security.check_csrf()

    @app.after_request
    def _security_headers(response):
        """Заголовки ответа, которые защищают страницы системы в браузере пользователя."""
        response.headers.setdefault("X-Content-Type-Options", "nosniff")    # тип содержимого не угадывается
        response.headers.setdefault("X-Frame-Options", "DENY")              # страницу нельзя встроить в чужой сайт
        response.headers.setdefault("Referrer-Policy", "same-origin")       # адреса страниц не уходят наружу
        response.headers.setdefault("Content-Security-Policy", CONTENT_SECURITY_POLICY)
        if request.endpoint != "static":
            # Страницы с данными не остаются в кэше браузера: после выхода из системы
            # их нельзя открыть кнопкой «Назад» на общем компьютере
            response.headers["Cache-Control"] = "no-store"
        return response

    @app.teardown_appcontext
    def _close_session(_error=None):
        # Возвращаем соединение с базой данных в пул после каждого запроса
        get_session_factory().remove()

    @app.context_processor
    def _template_globals():
        user = security.current_user()
        menu = [(endpoint, title, icon) for endpoint, title, icon, roles in MENU
                if user is not None and user.role.code in roles]
        return {"user": user, "menu": menu, "csrf_token": security.csrf_token, "class_names": CLASS_NAMES,
                "has_role": security.has_role, "now": datetime.now(),
                "active_menu": active_menu(request.endpoint or "")}

    app.jinja_env.filters["num"] = format_number
    app.jinja_env.filters["pct"] = lambda value: "—" if value is None else f"{value * 100:.1f}".replace(".", ",") + " %"
    app.jinja_env.filters["dt"] = lambda value: value.strftime("%d.%m.%Y %H:%M") if value else "—"

    @app.errorhandler(400)
    def _bad_request(error):
        return render_template("error.html", code=400, title="Некорректный запрос",
                               text=getattr(error, "description", "")), 400

    @app.errorhandler(403)
    def _forbidden(_error):
        return render_template("error.html", code=403, title="Доступ запрещён",
                               text="У вашей роли нет прав на выполнение этого действия."), 403

    @app.errorhandler(404)
    def _not_found(_error):
        return render_template("error.html", code=404, title="Страница не найдена",
                               text="Проверьте адрес или вернитесь на главную страницу."), 404

    @app.errorhandler(413)
    def _too_large(_error):
        return render_template("error.html", code=413, title="Файл слишком большой",
                               text="Размер загружаемого файла не должен превышать 64 МБ."), 413

    @app.errorhandler(500)
    def _server_error(_error):
        # Подробности ошибки Flask записывает в журнал сервера; пользователю показываем страницу на русском языке
        return render_template("error.html", code=500, title="Внутренняя ошибка",
                               text="Действие не выполнено из-за ошибки на сервере. Вернитесь на главную страницу; "
                                    "если ошибка повторяется, сообщите администратору."), 500

    # ------------------------------------------------ наборы страниц
    from webapp.views_admin import bp as admin_bp
    from webapp.views_auth import bp as auth_bp
    from webapp.views_data import bp as data_bp
    from webapp.views_feedback import bp as feedback_bp
    from webapp.views_forecast import bp as forecast_bp
    from webapp.views_main import bp as main_bp
    from webapp.views_models import bp as models_bp
    from webapp.views_reports import bp as reports_bp
    for blueprint in (auth_bp, main_bp, data_bp, models_bp, forecast_bp, reports_bp, feedback_bp, admin_bp):
        app.register_blueprint(blueprint)

    from webapp.services import bootstrap
    # Обучения, оборванные прошлой остановкой системы, отмечаются как несостоявшиеся
    bootstrap.close_interrupted_experiments()
    if bootstrap_data:
        bootstrap.ensure_default_dataset()
    return app
