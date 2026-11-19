# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Страницы входа в систему, личного кабинета и справки.
"""

import math
import re
from datetime import datetime, timedelta

from flask import Blueprint, flash, redirect, render_template, request, session, url_for
from sqlalchemy import func, select

from vuzmon import __version__
from vuzmon.db.models import AuditLog, Experiment, User
from vuzmon.db.session import get_session, hash_password
from webapp import security
from webapp.services.jobs import MODEL_TYPES

bp = Blueprint("auth", __name__)

MIN_PASSWORD_LENGTH = 8

# Защита от подбора пароля: после MAX_FAILED_LOGINS неудачных попыток подряд вход с этим логином
# с того же адреса приостанавливается, пока самая ранняя из них не станет старше LOCK_MINUTES минут
MAX_FAILED_LOGINS = 5
LOCK_MINUTES = 10


# Логин: латинские буквы, цифры, точка и подчёркивание. Проверяется именно состав символов:
# метод str.isalnum считает буквами и кириллицу, и такой логин проходил вопреки правилу
LOGIN_PATTERN = re.compile(r"[A-Za-z0-9_.]{3,64}")
# Адрес электронной почты: имя, знак @ и домен с точкой, без пробелов
EMAIL_PATTERN = re.compile(r"[^@\s]+@[^@\s]+\.[^@\s.]+")


def login_problem(login: str) -> str:
    """Проверка логина новой учётной записи. Возвращает текст замечания или пустую строку."""
    if LOGIN_PATTERN.fullmatch(login) is None:
        return "Логин: от 3 до 64 латинских букв, цифр, точек или подчёркиваний."
    return ""


def email_problem(email: str) -> str:
    """Проверка адреса электронной почты (пустой адрес допустим). Возвращает замечание или пустую строку."""
    if email and (len(email) > 128 or EMAIL_PATTERN.fullmatch(email) is None):
        return "Адрес электронной почты указан неверно (пример: name@example.ru)."
    return ""


def password_problem(password: str) -> str:
    """Проверка сложности пароля. Возвращает текст замечания или пустую строку."""
    if len(password) < MIN_PASSWORD_LENGTH:
        return f"Пароль должен содержать не менее {MIN_PASSWORD_LENGTH} символов."
    if password.isdigit() or password.isalpha():
        return "Пароль должен содержать и буквы, и цифры или специальные символы."
    return ""


def _failure_details(login_value: str) -> str:
    """Текст записи журнала о неудачной попытке входа (по нему же попытки подсчитываются)."""
    return f"Неудачная попытка входа: {login_value[:60]}"


def recent_failures(db, login_value: str, user) -> list:
    """
    Время недавних неудачных попыток входа с данным логином и с данного адреса.

    Учитываются попытки за последние LOCK_MINUTES минут, сделанные после
    последнего успешного входа: верный пароль обнуляет счётчик. Попытки
    считаются и для несуществующих логинов — по ответу системы нельзя понять,
    есть ли такая учётная запись.
    """
    since = datetime.now() - timedelta(minutes=LOCK_MINUTES)
    if user is not None:
        last_success = db.scalar(select(func.max(AuditLog.created_at)).where(
            AuditLog.user_id == user.id, AuditLog.action == "login"))
        if last_success is not None and last_success > since:
            since = last_success
    return db.scalars(select(AuditLog.created_at).where(
        AuditLog.action == "login_failed", AuditLog.details == _failure_details(login_value),
        AuditLog.ip_address == (request.remote_addr or ""), AuditLog.created_at > since)
        .order_by(AuditLog.created_at)).all()


@bp.route("/login", methods=["GET", "POST"])
def login():
    """Форма входа: проверка логина и пароля, создание сессии."""
    if security.current_user() is not None:
        return redirect(url_for("main.dashboard"))
    if request.method == "POST":
        db = get_session()
        login_value = request.form.get("login", "").strip()
        user = db.scalars(select(User).where(User.login == login_value)).first()
        failures = recent_failures(db, login_value, user)
        if len(failures) >= MAX_FAILED_LOGINS:
            # Пароль не проверяется вовсе: пока вход приостановлен, перебор ничего не даёт
            unlock_at = failures[-MAX_FAILED_LOGINS] + timedelta(minutes=LOCK_MINUTES)
            minutes = max(1, math.ceil((unlock_at - datetime.now()).total_seconds() / 60))
            security.audit("login_blocked", f"Вход приостановлен после {MAX_FAILED_LOGINS} неудачных попыток: "
                                            f"{login_value[:60]}")
            flash(f"Слишком много неудачных попыток входа. Повторите через {minutes} мин.", "error")
            return render_template("login.html"), 429
        if user is not None and user.is_active and security.verify_password(user, request.form.get("password", "")):
            # Новая сессия при входе защищает от подмены идентификатора сессии
            session.clear()
            session["user_id"] = user.id
            user.last_login_at = datetime.now()
            db.commit()
            security.audit("login", "Вход в систему", user_id=user.id)
            # Переходим только по внутренним адресам (защита от открытого перенаправления)
            return redirect(security.safe_next_url(request.args.get("next", "")) or url_for("main.dashboard"))
        security.audit("login_failed", _failure_details(login_value))
        left = MAX_FAILED_LOGINS - len(failures) - 1
        # О скорой приостановке предупреждаем заранее, когда попыток осталось мало
        flash("Неверный логин или пароль." + (f" Осталось попыток: {left}." if 0 < left <= 2 else ""), "error")
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
            email = request.form.get("email", "").strip()
            if email_problem(email):
                flash(email_problem(email), "error")
            else:
                user.full_name = request.form.get("full_name", "").strip()[:128]
                user.email = email
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
