# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Страница «Обратная связь»: пользователь любой роли отправляет обращение
администратору (вопрос, сообщение об ошибке, замечание к данным, предложение)
и видит состояние своих обращений и ответы на них. Обращения хранятся в
таблице feedback_messages; администратор отвечает на них в своей панели.
"""

from flask import Blueprint, flash, redirect, render_template, request, url_for
from sqlalchemy import select

from vuzmon.db.models import FeedbackMessage
from vuzmon.db.session import get_session
from webapp import security

bp = Blueprint("feedback", __name__, url_prefix="/feedback")

# Виды обращений и состояния их рассмотрения: код -> название
CATEGORIES = {"question": "Вопрос по работе с системой", "error": "Сообщение об ошибке",
              "data": "Замечание к данным мониторинга", "proposal": "Предложение по развитию"}
STATUSES = {"new": "новое", "in_work": "в работе", "closed": "закрыто"}

SUBJECT_MIN, SUBJECT_MAX = 5, 160
TEXT_MIN, TEXT_MAX = 10, 2000


def form_problems(category: str, subject: str, text: str) -> list:
    """Проверка полей формы. Возвращает список замечаний (пустой, если всё верно)."""
    problems = []
    if category not in CATEGORIES:
        problems.append("Выберите вид обращения.")
    if not SUBJECT_MIN <= len(subject) <= SUBJECT_MAX:
        problems.append(f"Тема: от {SUBJECT_MIN} до {SUBJECT_MAX} символов.")
    if not TEXT_MIN <= len(text) <= TEXT_MAX:
        problems.append(f"Текст обращения: от {TEXT_MIN} до {TEXT_MAX} символов.")
    return problems


@bp.route("/", methods=["GET", "POST"])
@security.login_required
def index():
    """Форма обращения и список собственных обращений пользователя."""
    db = get_session()
    user = security.current_user()
    form = {"category": "question", "subject": "", "text": ""}
    if request.method == "POST":
        form = {key: request.form.get(key, "").strip() for key in form}
        problems = form_problems(**form)
        if not problems:
            message = FeedbackMessage(user_id=user.id, **form)
            db.add(message)
            db.commit()
            security.audit("feedback_send", f"Обращение № {message.id}: {form['subject']}")
            flash("Обращение отправлено администратору. Ответ появится на этой странице.", "success")
            return redirect(url_for("feedback.index"))
        # При ошибке форма показывается снова с введённым текстом
        for problem in problems:
            flash(problem, "error")

    # Пользователь видит только собственные обращения
    rows = db.scalars(select(FeedbackMessage).where(FeedbackMessage.user_id == user.id)
                      .order_by(FeedbackMessage.id.desc())).all()
    return render_template("feedback.html", crumbs=[("Обратная связь", None)], rows=rows, form=form,
                           categories=CATEGORIES, statuses=STATUSES,
                           limits={"subject": (SUBJECT_MIN, SUBJECT_MAX), "text": (TEXT_MIN, TEXT_MAX)})
