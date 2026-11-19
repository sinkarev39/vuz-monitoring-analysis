# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Панель администратора: пользователи, обращения пользователей (обратная связь),
журнал действий, настройки системы, сведения о состоянии базы данных и файлов.
"""

import platform
import sys
from datetime import datetime, timedelta
from importlib import metadata

from flask import Blueprint, abort, flash, redirect, render_template, request, send_file, url_for
from sqlalchemy import func, select, text

from vuzmon import __version__, config
from vuzmon.db.models import AuditLog, Base, FeedbackMessage, Role, Setting, University, User
from vuzmon.db.session import get_engine, get_session, hash_password
from webapp import security
from webapp.forms import to_int
from webapp.views_auth import password_problem
from webapp.views_feedback import CATEGORIES, STATUSES, TEXT_MAX

bp = Blueprint("admin", __name__, url_prefix="/admin")

# Журнал показывается не более чем за десять лет: большее число дней не имеет смысла
MAX_AUDIT_DAYS = 3650

# Вкладки панели администратора
TABS = [("admin.users", "Пользователи"), ("admin.feedback", "Обращения"),
        ("admin.audit", "Журнал действий"), ("admin.settings", "Настройки"),
        ("admin.system", "Состояние системы"), ("admin.roles", "Роли и права")]

# Библиотеки, версии которых показываются на странице состояния системы
LIBRARIES = ["flask", "sqlalchemy", "torch", "numpy", "pandas", "scikit-learn", "matplotlib",
             "requests", "beautifulsoup4", "lxml", "python-docx", "openpyxl"]

# Правила проверки настроек: ключ -> (вид значения, наименьшее, наибольшее).
# Для текста границы — длина строки, для чисел — само значение.
SETTING_RULES = {
    "organization_name": ("text", 3, 255),
    "home_university_id": ("university", None, None),
    "base_year": ("int", 2017, 2100),
    "records_per_page": ("int", 5, 200),
    "risk_threshold": ("float", 0.05, 0.95),
    "anomaly_quantile": ("float", 0.5, 0.999),
    "collector_interval": ("float", 0.1, 10.0),
}


def _plain_number(value) -> str:
    """Число в записи для сообщений и хранения: без лишних нулей, с точкой."""
    return f"{value:g}"


def check_setting(db, key: str, raw: str) -> tuple:
    """
    Проверяет значение настройки. Возвращает пару (значение для сохранения,
    замечание): если замечание не пустое, значение сохранять нельзя.
    """
    value = raw.strip()[:255]
    kind, low, high = SETTING_RULES.get(key, ("text", 0, 255))
    if kind == "text":
        if not low <= len(value) <= high:
            return value, f"введите текст длиной от {low} до {high} символов"
        return value, ""
    if kind == "university":
        if value == "":
            return value, ""                      # «своя» организация может быть не задана
        if not value.isdigit() or db.get(University, int(value)) is None:
            return value, "организации с таким кодом нет в базе данных (код указан в карточке организации)"
        return str(int(value)), ""
    limits = f"от {_plain_number(low).replace('.', ',')} до {_plain_number(high).replace('.', ',')}"
    try:
        number = (int if kind == "int" else float)(value.replace(",", ".").replace(" ", ""))
    except ValueError:
        return value, f"введите {'целое число' if kind == 'int' else 'число'} {limits}"
    if number != number or not low <= number <= high:          # первое условие отсекает NaN
        return value, f"значение должно быть {limits}"
    return _plain_number(number) if kind == "float" else str(number), ""


# Матрица прав: действие -> роли, которым оно доступно
PERMISSIONS = [
    ("Просмотр обзора, организаций, прогноза, нетипичных профилей", ("admin", "analyst", "viewer")),
    ("Сравнение организаций, расчёт «что будет, если…»", ("admin", "analyst", "viewer")),
    ("Выгрузка справок и таблиц прогноза (Word, Excel)", ("admin", "analyst", "viewer")),
    ("Отправка обращений администратору (обратная связь)", ("admin", "analyst", "viewer")),
    ("Заметки к организациям", ("admin", "analyst")),
    ("Сбор, загрузка и выгрузка наборов данных", ("admin", "analyst")),
    ("Обучение моделей, подбор гиперпараметров, выбор рабочей модели", ("admin", "analyst")),
    ("Построение и обновление прогноза", ("admin", "analyst")),
    ("Управление пользователями и ролями", ("admin",)),
    ("Ответы на обращения пользователей", ("admin",)),
    ("Просмотр и выгрузка журнала действий", ("admin",)),
    ("Изменение настроек системы, резервная копия базы данных", ("admin",)),
]


def _render(template: str, title: str, **context):
    return render_template(template, crumbs=[("Администрирование", url_for("admin.users")), (title, None)],
                           tabs=TABS, **context)


# --------------------------------------------------------------------------
# Пользователи
# --------------------------------------------------------------------------
@bp.route("/users")
@security.roles_required(*security.ROLE_ADMIN)
def users():
    """Список учётных записей."""
    db = get_session()
    rows = db.scalars(select(User).order_by(User.login)).all()
    return _render("admin_users.html", "Пользователи", rows=rows)


@bp.route("/users/new", methods=["GET", "POST"])
@bp.route("/users/<int:user_id>", methods=["GET", "POST"])
@security.roles_required(*security.ROLE_ADMIN)
def user_form(user_id: int = None):
    """Создание и изменение учётной записи."""
    db = get_session()
    item = db.get(User, user_id) if user_id else None
    if user_id and item is None:
        abort(404)
    roles = db.scalars(select(Role).order_by(Role.id)).all()

    if request.method == "POST":
        form = request.form
        login = form.get("login", "").strip()
        password = form.get("password", "")
        role = db.get(Role, to_int(form.get("role_id")))      # нечисловой код роли — то же, что роль не выбрана
        errors = []
        if item is None:
            if not (3 <= len(login) <= 64) or not login.replace("_", "").replace(".", "").isalnum():
                errors.append("Логин: от 3 до 64 латинских букв, цифр, точек или подчёркиваний.")
            elif db.scalars(select(User).where(User.login == login)).first() is not None:
                errors.append("Пользователь с таким логином уже существует.")
            if not password:
                errors.append("Задайте пароль для новой учётной записи.")
        if password and password_problem(password):
            errors.append(password_problem(password))
        if role is None:
            errors.append("Выберите роль.")
        is_active = bool(form.get("is_active"))
        me = security.current_user()
        if item is not None and item.id == me.id and (not is_active or (role and role.code != "admin")):
            errors.append("Нельзя отключить собственную учётную запись или снять с себя роль администратора.")
        if errors:
            for message in errors:
                flash(message, "error")
        else:
            if item is None:
                item = User(login=login, password_hash=hash_password(password))
                db.add(item)
                action = "user_create"
            else:
                action = "user_update"
                if password:
                    item.password_hash = hash_password(password)
            item.full_name = form.get("full_name", "").strip()[:128]
            item.email = form.get("email", "").strip()[:128]
            item.position = form.get("position", "").strip()[:128]
            item.role_id, item.is_active = role.id, is_active
            db.commit()
            security.audit(action, f"Учётная запись «{item.login}»: роль {role.title}, "
                                   f"{'активна' if is_active else 'отключена'}")
            flash("Учётная запись сохранена.", "success")
            return redirect(url_for("admin.users"))
    return _render("admin_user_form.html", "Учётная запись", item=item, roles=roles)


# --------------------------------------------------------------------------
# Обращения пользователей (обратная связь)
# --------------------------------------------------------------------------
@bp.route("/feedback")
@security.roles_required(*security.ROLE_ADMIN)
def feedback():
    """Обращения пользователей с отбором по состоянию; новые показываются первыми."""
    db = get_session()
    status = request.args.get("status", "")
    query = select(FeedbackMessage)
    if status in STATUSES:
        query = query.where(FeedbackMessage.status == status)
    # Порядок: новые, затем взятые в работу, затем закрытые; внутри группы — свежие сверху
    order = {code: number for number, code in enumerate(STATUSES)}
    rows = sorted(db.scalars(query).all(), key=lambda row: (order.get(row.status, 9), -row.id))
    counts = dict(db.execute(select(FeedbackMessage.status, func.count()).group_by(FeedbackMessage.status)).all())
    return _render("admin_feedback.html", "Обращения", rows=rows, counts=counts, status=status,
                   categories=CATEGORIES, statuses=STATUSES, text_max=TEXT_MAX)


@bp.route("/feedback/<int:message_id>", methods=["POST"])
@security.roles_required(*security.ROLE_ADMIN)
def feedback_reply(message_id: int):
    """Ответ на обращение и изменение его состояния."""
    db = get_session()
    item = db.get(FeedbackMessage, message_id)
    if item is None:
        abort(404)
    reply = request.form.get("reply", "").strip()[:TEXT_MAX]
    status = request.form.get("status", "")
    if status not in STATUSES:
        flash("Выберите состояние обращения.", "error")
    elif status == "closed" and not reply:
        flash("Чтобы закрыть обращение, напишите ответ пользователю.", "error")
    else:
        if reply != item.reply:
            # Автор и время ответа обновляются только при изменении текста ответа
            item.reply = reply
            item.replied_by = security.current_user().id if reply else None
            item.replied_at = datetime.now() if reply else None
        item.status = status
        db.commit()
        security.audit("feedback_reply", f"Обращение № {item.id} от «{item.author.login}»: {STATUSES[status]}")
        flash(f"Обращение № {item.id} сохранено.", "success")
    return redirect(url_for("admin.feedback", status=request.args.get("status") or None))


# --------------------------------------------------------------------------
# Журнал действий
# --------------------------------------------------------------------------
@bp.route("/audit")
@security.roles_required(*security.ROLE_ADMIN)
def audit():
    """Журнал действий с фильтрами по пользователю, действию и периоду."""
    db = get_session()
    args = request.args
    query = select(AuditLog)
    if args.get("user", "").isdigit():
        query = query.where(AuditLog.user_id == int(args["user"]))
    if args.get("action"):
        query = query.where(AuditLog.action == args["action"])
    days = to_int(args.get("days"), 0, low=0, high=MAX_AUDIT_DAYS)
    if days:
        query = query.where(AuditLog.created_at >= datetime.now() - timedelta(days=days))
    total = db.scalar(select(func.count()).select_from(query.subquery()))
    size = 50
    pages = max(1, (total + size - 1) // size)
    page = min(max(1, int(args.get("page", "1")) if args.get("page", "1").isdigit() else 1), pages)
    rows = db.scalars(query.order_by(AuditLog.created_at.desc()).offset((page - 1) * size).limit(size)).all()
    return _render("admin_audit.html", "Журнал действий", rows=rows, total=total, page=page, pages=pages,
                   filters={k: v for k, v in args.items() if k != "page" and v},
                   users=db.scalars(select(User).order_by(User.login)).all(),
                   actions=db.scalars(select(AuditLog.action).distinct().order_by(AuditLog.action)).all())


# --------------------------------------------------------------------------
# Настройки
# --------------------------------------------------------------------------
@bp.route("/settings", methods=["GET", "POST"])
@security.roles_required(*security.ROLE_ADMIN)
def settings():
    """Изменение настроек системы."""
    db = get_session()
    # В панели изменяются только настройки с правилами проверки. Служебные значения (например, код
    # рабочего набора данных) меняются действиями в разделах системы, где выполняются нужные проверки
    rows = [row for row in db.scalars(select(Setting).order_by(Setting.key)) if row.key in SETTING_RULES]
    entered = {}
    if request.method == "POST":
        # Сначала проверяются все поля; при любом замечании не сохраняется ничего
        checked, problems = {}, []
        for row in rows:
            if row.key not in request.form:
                continue                          # поле не передано — настройка остаётся прежней
            entered[row.key] = request.form[row.key].strip()[:255]
            checked[row.key], problem = check_setting(db, row.key, request.form[row.key])
            if problem:
                problems.append(f"«{row.title}»: {problem}.")
        if not problems:
            changed = []
            for row in rows:
                if row.key in checked and checked[row.key] != row.value:
                    changed.append(f"{row.key}: «{row.value}» → «{checked[row.key]}»")
                    row.value = checked[row.key]
            db.commit()
            if changed:
                security.audit("settings_update", "; ".join(changed))
            flash("Настройки сохранены." if changed else "Изменений нет.", "success" if changed else "info")
            return redirect(url_for("admin.settings"))
        for problem in problems:
            flash(problem, "error")
        flash("Настройки не сохранены: исправьте отмеченные значения.", "error")
    home = db.get(Setting, "home_university_id")
    home_university = db.get(University, int(home.value)) if home and home.value.isdigit() else None
    # При ошибке форма показывается снова с введёнными значениями
    return _render("admin_settings.html", "Настройки", rows=rows, home_university=home_university, entered=entered)


# --------------------------------------------------------------------------
# Состояние системы
# --------------------------------------------------------------------------
def _folder_size(path) -> tuple:
    """Число файлов и суммарный размер каталога в байтах."""
    files = [f for f in path.rglob("*") if f.is_file()] if path.exists() else []
    return len(files), sum(f.stat().st_size for f in files)


@bp.route("/system")
@security.roles_required(*security.ROLE_ADMIN)
def system():
    """Сведения о базе данных, файлах и версиях библиотек."""
    db = get_session()
    tables = [{"name": table.name, "rows": db.scalar(select(func.count()).select_from(table))}
              for table in Base.metadata.sorted_tables]
    libraries = []
    for name in LIBRARIES:
        try:
            libraries.append((name, metadata.version(name)))
        except metadata.PackageNotFoundError:
            libraries.append((name, "не установлена"))
    folders = [("База данных", config.DB_PATH.parent, config.DB_PATH.stat().st_size if config.DB_PATH.exists() else 0, 1),
               ("Набор данных (CSV)", config.MONITORING_DIR, *reversed(_folder_size(config.MONITORING_DIR))),
               ("Файлы моделей", config.MODELS_DIR, *reversed(_folder_size(config.MODELS_DIR))),
               ("Кэш страниц сайта", config.CACHE_DIR, *reversed(_folder_size(config.CACHE_DIR)))]
    return _render("admin_system.html", "Состояние системы", tables=tables, libraries=libraries,
                   folders=[{"title": t, "path": str(p), "size": s, "files": n} for t, p, s, n in folders],
                   python=sys.version.split()[0], os_name=f"{platform.system()} {platform.release()}",
                   version=__version__, dialect=get_engine().dialect.name)


@bp.route("/system/backup")
@security.roles_required(*security.ROLE_ADMIN)
def backup():
    """Резервная копия базы данных (файл SQLite)."""
    if get_engine().dialect.name != "sqlite" or not config.DB_PATH.exists():
        flash("Резервная копия файлом доступна только для базы данных SQLite.", "error")
        return redirect(url_for("admin.system"))
    security.audit("db_backup", "Скачана резервная копия базы данных")
    return send_file(config.DB_PATH, as_attachment=True,
                     download_name=f"vuzmon_backup_{datetime.now():%Y%m%d_%H%M}.db")


@bp.route("/system/optimize", methods=["POST"])
@security.roles_required(*security.ROLE_ADMIN)
def optimize():
    """Сжатие файла базы данных (команда VACUUM) после массовых загрузок."""
    engine = get_engine()
    if engine.dialect.name == "sqlite":
        get_session().commit()
        with engine.connect().execution_options(isolation_level="AUTOCOMMIT") as connection:
            connection.execute(text("VACUUM"))
        security.audit("db_optimize", "Выполнено сжатие базы данных")
        flash("База данных сжата.", "success")
    return redirect(url_for("admin.system"))


@bp.route("/roles")
@security.roles_required(*security.ROLE_ADMIN)
def roles():
    """Справочник ролей и матрица прав доступа."""
    db = get_session()
    rows = db.scalars(select(Role).order_by(Role.id)).all()
    counts = dict(db.execute(select(User.role_id, func.count()).group_by(User.role_id)).all())
    return _render("admin_roles.html", "Роли и права", rows=rows, counts=counts, permissions=PERMISSIONS)
