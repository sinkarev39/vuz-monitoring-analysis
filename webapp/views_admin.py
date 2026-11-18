# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Панель администратора: пользователи, журнал действий, настройки системы,
сведения о состоянии базы данных и файлов.
"""

import platform
import sys
from datetime import datetime, timedelta
from importlib import metadata

from flask import Blueprint, abort, flash, redirect, render_template, request, send_file, url_for
from sqlalchemy import func, select, text

from vuzmon import __version__, config
from vuzmon.db.models import AuditLog, Base, Role, Setting, University, User
from vuzmon.db.session import get_engine, get_session, hash_password
from webapp import security
from webapp.views_auth import password_problem

bp = Blueprint("admin", __name__, url_prefix="/admin")

# Вкладки панели администратора
TABS = [("admin.users", "Пользователи"), ("admin.audit", "Журнал действий"),
        ("admin.settings", "Настройки"), ("admin.system", "Состояние системы"),
        ("admin.roles", "Роли и права")]

# Библиотеки, версии которых показываются на странице состояния системы
LIBRARIES = ["flask", "sqlalchemy", "torch", "numpy", "pandas", "scikit-learn", "matplotlib",
             "requests", "beautifulsoup4", "lxml", "python-docx", "openpyxl"]

# Матрица прав: действие -> роли, которым оно доступно
PERMISSIONS = [
    ("Просмотр обзора, организаций, прогноза, нетипичных профилей", ("admin", "analyst", "viewer")),
    ("Сравнение организаций, расчёт «что будет, если…»", ("admin", "analyst", "viewer")),
    ("Выгрузка справок и таблиц прогноза (Word, Excel)", ("admin", "analyst", "viewer")),
    ("Заметки к организациям", ("admin", "analyst")),
    ("Сбор, загрузка и выгрузка наборов данных", ("admin", "analyst")),
    ("Обучение моделей, подбор гиперпараметров, выбор рабочей модели", ("admin", "analyst")),
    ("Построение и обновление прогноза", ("admin", "analyst")),
    ("Управление пользователями и ролями", ("admin",)),
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
        role = db.get(Role, int(form.get("role_id", "0") or 0))
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
    if args.get("days", "").isdigit():
        query = query.where(AuditLog.created_at >= datetime.now() - timedelta(days=int(args["days"])))
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
    rows = db.scalars(select(Setting).where(Setting.title != "").order_by(Setting.key)).all()
    if request.method == "POST":
        changed = []
        for row in rows:
            value = request.form.get(row.key, "").strip()[:255]
            if value != row.value:
                changed.append(f"{row.key}: «{row.value}» → «{value}»")
                row.value = value
        db.commit()
        if changed:
            security.audit("settings_update", "; ".join(changed))
        flash("Настройки сохранены." if changed else "Изменений нет.", "success" if changed else "info")
        return redirect(url_for("admin.settings"))
    home = db.get(Setting, "home_university_id")
    home_university = db.get(University, int(home.value)) if home and home.value.isdigit() else None
    return _render("admin_settings.html", "Настройки", rows=rows, home_university=home_university)


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
