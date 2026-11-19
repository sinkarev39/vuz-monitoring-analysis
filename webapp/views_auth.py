# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Страницы входа в систему, личного кабинета и справки.
"""

from datetime import datetime

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from sqlalchemy import select

from vuzmon import __version__
from vuzmon.db.models import AuditLog, Experiment, User
from vuzmon.db.session import get_session, hash_password
from webapp import security
from webapp.services.jobs import MODEL_TYPES

bp = Blueprint("auth", __name__)

MIN_PASSWORD_LENGTH = 8


def password_problem(password: str) -> str:
    """Проверка сложности пароля. Возвращает текст замечания или пустую строку."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Пароль должен содержать не менее {MIN_PASSWORD_LENGTH} символов."
    if password.isdigit() or password.isalpha():
        return "Пароль должен содержать и буквы, и цифры или специальные символы."
    return ""


@bp.route("/login", methods=["GET", "POST"])
def login():
    """Форма входа: проверка логина и пароля, создание сессии."""
    if security.current_user() is not None:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        db = get_session()
        login_value = request.form.get("login", "").strip()
        user = db.scalars(select(User).where(User.login == login_value)).first()
        if user is not None and user.is_active and security.verify_password(user, request.form.get("password", "")):
            # Новая сессия при входе защищает от подмены идентификатора сессии
            session.clear()
            session["user_id"] = user.id
            user.last_login_at = datetime.now()
            db.commit()
            security.audit("login", "Вход в систему", user_id=user.id)
            # Переходим только по внутренним адресам (защита от открытого перенаправления)
            return redirect(security.safe_next_url(request.args.get("next", "")) or url_for("main.dashboard"))
        security.audit("login_failed", f"Неудачная попытка входа: {login_value[:60]}")
        flash("Неверный логин или пароль.", "error")
    return render_template("login.html")


@bp.route("/logout", methods=["POST"])
def logout():
    """Выход из системы."""
    if security.current_user() is not None:
        security.audit("logout", "Выход из системы")
    session.clear()
    return redirect(url_for("auth.login"))


@bp.route("/profile", methods=["GET", "POST"])
@security.login_required
def profile():
    """Личный кабинет: сведения о пользователе, смена пароля, собственные действия."""
    db = get_session()
    user = security.current_user()
    if request.method == "POST":
        action = request.form.get("action")
        if action == "details":
            user.full_name = request.form.get("full_name", "").strip()[:128]
            user.email = request.form.get("email", "").strip()[:128]
            user.position = request.form.get("position", "").strip()[:128]
            db.commit()
            security.audit("profile_update", "Изменены сведения в личном кабинете")
            flash("Сведения сохранены.", "success")
        elif action == "password":
            new_password = request.form.get("new_password", "")
            problem = password_problem(new_password)
            if not security.verify_password(user, request.form.get("old_password", "")):
                flash("Текущий пароль указан неверно.", "error")
            elif new_password != request.form.get("repeat_password", ""):
                flash("Новый пароль и его повтор не совпадают.", "error")
            elif problem:
                flash(problem, "error")
            else:
                user.password_hash = hash_password(new_password)
                db.commit()
                security.audit("password_change", "Пароль изменён пользователем")
                flash("Пароль изменён.", "success")
        return redirect(url_for("auth.profile"))

    actions = db.scalars(select(AuditLog).where(AuditLog.user_id == user.id)
                         .order_by(AuditLog.created_at.desc()).limit(15)).all()
    experiments = db.scalars(select(Experiment).where(Experiment.created_by == user.id)
                             .order_by(Experiment.id.desc()).limit(10)).all()
    return render_template("profile.html", crumbs=[("Личный кабинет", None)], actions=actions,
                           experiments=experiments, model_types=MODEL_TYPES)


@bp.route("/help")
@security.login_required
def help_page():
    """Справка по системе со сведениями об авторе."""
    return render_template("help.html", crumbs=[("Справка", None)], version=__version__)
