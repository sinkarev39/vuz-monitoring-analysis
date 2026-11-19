# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Раздел «Отчёты»: формирование документов Word и Excel по данным системы.
"""

import json

from flask import Blueprint, abort, flash, redirect, render_template, request, send_file, url_for
from sqlalchemy import select

from vuzmon.db.models import AnalystNote, AuditLog, Experiment, FederalDistrict, University
from vuzmon.db.session import get_number_setting, get_session, get_setting
from webapp import security
from webapp.services import exporters, jobs, queries, store
from webapp.views_main import explain_prediction

bp = Blueprint("reports", __name__, url_prefix="/reports")

DOCX = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _user_name() -> str:
    user = security.current_user()
    return user.full_name or user.login


def _prepared(db):
    dataset = store.active_dataset(db)
    if dataset is None or not store.dataset_is_complete(dataset.location):
        return None
    return store.prepared(dataset.location)


def _risk_threshold(db) -> float:
    """Порог вероятности зоны риска из настроек системы."""
    return get_number_setting(db, "risk_threshold", 0.5, 0.05, 0.95)


def _organization(db) -> str:
    """Название организации для подписи формируемых документов (настройка organization_name)."""
    return get_setting(db, "organization_name")


def _forecast_rows(db, args) -> tuple:
    """Строки прогноза с учётом фильтров страницы «Прогноз»."""
    predictions = queries.latest_predictions(db)
    prepared = _prepared(db)
    current = queries.current_classes(prepared) if prepared else {}
    rows = queries.forecast_rows(db, args, predictions, current, _risk_threshold(db))
    rows.sort(key=lambda row: row["prediction"]["probabilities"][0], reverse=True)
    return rows, predictions


@bp.route("/")
@security.login_required
def index():
    """Перечень доступных отчётов."""
    db = get_session()
    predictions = queries.latest_predictions(db)
    first = next(iter(predictions.values()), None)
    experiments = db.scalars(select(Experiment).where(Experiment.status == "done")
                             .order_by(Experiment.id.desc()).limit(20)).all()
    return render_template("reports.html", crumbs=[("Отчёты", None)], experiments=experiments,
                           target_year=first["target_year"] if first else None,
                           districts=db.scalars(select(FederalDistrict).order_by(FederalDistrict.name)).all(),
                           model_types=jobs.MODEL_TYPES)


@bp.route("/find-university")
@security.login_required
def find_university():
    """Переход к справке по организации по части её названия (форма на странице отчётов)."""
    db = get_session()
    search = request.args.get("q", "").strip()
    item = db.scalars(select(University).where(queries.name_filter(search))
                      .order_by(University.is_branch, University.short_name)).first() if search else None
    if item is None:
        flash("Организация с таким названием не найдена.", "error")
        return redirect(url_for("reports.index"))
    return redirect(url_for("reports.university_docx", university_id=item.id))


@bp.route("/university/<int:university_id>.docx")
@security.login_required
def university_docx(university_id: int):
    """Аналитическая справка по организации в формате Word."""
    db = get_session()
    item = db.get(University, university_id)
    if item is None:
        abort(404)
    values = queries.university_values(db, university_id)
    table = queries.main_indicator_table(values, queries.university_medians(db, item), item.is_branch) \
        if not values.empty else None
    prediction = queries.latest_predictions(db).get(university_id)
    explanation = explain_prediction(db, _prepared(db), university_id) if prediction else {}
    notes = db.scalars(select(AnalystNote).where(AnalystNote.university_id == university_id)
                       .order_by(AnalystNote.created_at.desc())).all()
    buffer = exporters.university_docx(item, table, prediction, explanation, notes, _user_name(), _organization(db))
    security.audit("report_university", f"Сформирована справка по организации {item.short_name}")
    return send_file(buffer, mimetype=DOCX, as_attachment=True, download_name=f"spravka_{item.id}.docx")


@bp.route("/forecast.xlsx")
@security.login_required
def forecast_xlsx():
    """Таблица прогноза в формате Excel (учитывает фильтры)."""
    db = get_session()
    rows, predictions = _forecast_rows(db, request.args)
    if not predictions:
        flash("Прогноз ещё не построен.", "error")
        return redirect(url_for("reports.index"))
    first = next(iter(predictions.values()))
    buffer = exporters.forecast_xlsx(rows, first["base_year"], first["target_year"])
    security.audit("report_forecast", f"Выгружен прогноз в Excel: строк {len(rows)}")
    return send_file(buffer, mimetype=XLSX, as_attachment=True,
                     download_name=f"prognoz_{first['target_year']}.xlsx")


@bp.route("/summary.docx")
@security.login_required
def summary_docx():
    """Сводная аналитическая записка по прогнозу в формате Word."""
    db = get_session()
    rows, predictions = _forecast_rows(db, {})
    if not predictions:
        flash("Прогноз ещё не построен.", "error")
        return redirect(url_for("reports.index"))
    first = next(iter(predictions.values()))
    counts = [sum(1 for row in rows if row["prediction"]["predicted_class"] == label) for label in range(3)]
    districts = {}
    for row in rows:
        name = row["item"].region.district.name
        districts.setdefault(name, [0, 0, 0])[row["prediction"]["predicted_class"]] += 1
    by_district = [{"name": name, "counts": values} for name, values in sorted(districts.items())]

    def movers(direction: int) -> list:
        selected = [row for row in rows if row["change"] * direction > 0]
        selected.sort(key=lambda row: max(row["prediction"]["probabilities"]), reverse=True)
        return [{"name": row["item"].short_name, "region": row["item"].region.name, "now": row["current"],
                 "next": row["prediction"]["predicted_class"],
                 "confidence": max(row["prediction"]["probabilities"])} for row in selected[:25]]

    experiment = store.active_experiment(db, "mlp")
    model_line = ""
    if experiment is not None:
        test = json.loads(experiment.metrics_json or "{}").get("test")
        model_line = f"Использована модель №{experiment.id} «{experiment.name}»" + (
            f"; доля верных прогнозов на проверочном годе — {test['accuracy'] * 100:.1f} %.".replace(".", ",", 1)
            if test else ".")
    threshold = _risk_threshold(db)
    buffer = exporters.summary_docx(first["target_year"], counts, by_district, movers(-1), movers(1),
                                    model_line, _user_name(), _organization(db),
                                    risk=(threshold, queries.risk_count(predictions, threshold)))
    security.audit("report_summary", "Сформирована сводная записка по прогнозу")
    return send_file(buffer, mimetype=DOCX, as_attachment=True,
                     download_name=f"svodnaya_zapiska_{first['target_year']}.docx")


@bp.route("/experiment/<int:experiment_id>.docx")
@security.roles_required(*security.ROLE_ANALYST)
def experiment_docx(experiment_id: int):
    """Протокол обучения модели в формате Word."""
    db = get_session()
    experiment = db.get(Experiment, experiment_id)
    if experiment is None:
        abort(404)
    history = [{"epoch": e.epoch, "train_loss": e.train_loss, "val_loss": e.val_loss,
                "train_metric": e.train_metric, "val_metric": e.val_metric} for e in experiment.epochs]
    buffer = exporters.experiment_docx(
        experiment, json.loads(experiment.metrics_json or "{}"), json.loads(experiment.params_json or "{}"),
        history, jobs.MODEL_TYPES.get(experiment.model_type, experiment.model_type), _user_name(),
        _organization(db))
    security.audit("report_experiment", f"Сформирован протокол обучения эксперимента №{experiment.id}")
    return send_file(buffer, mimetype=DOCX, as_attachment=True,
                     download_name=f"protokol_obucheniya_{experiment.id}.docx")


@bp.route("/audit.xlsx")
@security.roles_required(*security.ROLE_ADMIN)
def audit_xlsx():
    """Журнал действий пользователей в формате Excel (только администратор)."""
    db = get_session()
    rows = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(5000)).all()
    security.audit("report_audit", "Выгружен журнал действий")
    return send_file(exporters.audit_xlsx(rows), mimetype=XLSX, as_attachment=True,
                     download_name="zhurnal_deystviy.xlsx")
