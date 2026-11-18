# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль security — аутентификация, разграничение доступа и журнал действий.

  * текущий пользователь определяется по идентификатору в подписанной сессии;
  * декораторы login_required и roles_required закрывают страницы от
    неавторизованных пользователей и пользователей без нужной роли;
  * все формы защищены от подделки межсайтовых запросов (CSRF) одноразовым
    для сессии токеном;
  * значимые действия записываются в журнал (таблица audit_log).
"""

import secrets
from functools import wraps

from flask import abort, flash, g, redirect, request, session, url_for
from werkzeug.security import check_password_hash

from vuzmon.db.models import AuditLog, User
from vuzmon.db.session import get_session

# Уровни доступа: какие роли могут выполнять действие
ROLE_ADMIN = ("admin",)
ROLE_ANALYST = ("admin", "analyst")          # работа с данными и моделями
ROLE_ANY = ("admin", "analyst", "viewer")    # просмотр


def current_user():
    """Пользователь, выполнивший вход (или None). Результат кэшируется на время запроса."""
    if "user" not in g:
        user_id = session.get("user_id")
        user = get_session().get(User, user_id) if user_id else None
        # Отключённая учётная запись теряет доступ даже при действующей сессии
        g.user = user if user is not None and user.is_active else None
    return g.user


def has_role(*codes) -> bool:
    """Есть ли у текущего пользователя одна из перечисленных ролей."""
    user = current_user()
    return user is not None and user.role.code in codes


def login_required(view):
    """Декоратор: страница доступна только после входа в систему."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        if current_user() is None:
            flash("Для продолжения работы войдите в систему.", "warning")
            return redirect(url_for("auth.login", next=request.path))
        return view(*args, **kwargs)
    return wrapper


def roles_required(*codes):
    """Декоратор: страница доступна только пользователям с указанными ролями."""
    def decorator(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            if current_user() is None:
                flash("Для продолжения работы войдите в систему.", "warning")
                return redirect(url_for("auth.login", next=request.path))
            if not has_role(*codes):
                abort(403)
            return view(*args, **kwargs)
        return wrapper
    return decorator


def verify_password(user: User, password: str) -> bool:
    """Сверяет введённый пароль с хешем из базы данных."""
    return user is not None and check_password_hash(user.password_hash, password)


def audit(action: str, details: str = "", user_id: int = None) -> None:
    """Добавляет запись в журнал действий пользователей."""
    db = get_session()
    user = current_user() if user_id is None else None
    db.add(AuditLog(user_id=user_id or (user.id if user else None), action=action, details=details[:2000],
                    ip_address=request.remote_addr or "" if request else ""))
    db.commit()


# ------------------------------------------------------------------ CSRF
def csrf_token() -> str:
    """Токен защиты форм; создаётся при первом обращении и живёт вместе с сессией."""
    if "_csrf" not in session:
        session["_csrf"] = secrets.token_hex(16)
    return session["_csrf"]


def check_csrf() -> None:
    """Проверка токена для всех изменяющих запросов (вызывается перед обработчиком)."""
    if request.method in ("POST", "PUT", "DELETE"):
        sent = request.form.get("_csrf") or request.headers.get("X-CSRF-Token", "")
        if not sent or not secrets.compare_digest(sent, session.get("_csrf", "")):
            abort(400, description="Форма устарела. Обновите страницу и повторите действие.")
