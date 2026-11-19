# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Основные страницы системы: обзор, реестр организаций, карточка организации,
сравнение организаций.
"""

import json

import numpy as np
from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from sqlalchemy import func, select

from vuzmon.config import MAIN_INDICATOR_TITLES
from vuzmon.db.models import (AnalystNote, AuditLog, FederalDistrict, FounderGroup, Prediction, Region,
                              University)
from vuzmon.db.session import get_session, get_setting
from vuzmon.ml import explain, pipeline, plots, trainer
from vuzmon.ml.dataset import CLASS_NAMES, target_indicators
from webapp import security
from webapp.services import queries, store

bp = Blueprint("main", __name__)


def _prepared_or_none(db):
    """Подготовленная выборка активного набора данных (None, если данных нет)."""
    dataset = store.active_dataset(db)
    if dataset is None or not store.dataset_is_complete(dataset.location):
        return None
    return store.prepared(dataset.location)


def _page_size(db) -> int:
    value = get_setting(db, "records_per_page", "25")
    return max(5, min(200, int(value))) if value.isdigit() else 25


# --------------------------------------------------------------------------
# Обзор
# --------------------------------------------------------------------------
@bp.route("/")
@security.login_required
def dashboard():
    """Главная страница: ключевые цифры, распределение по классам, состояние моделей."""
    db = get_session()
    summary = queries.database_summary(db)
    prepared = _prepared_or_none(db)
    predictions = queries.latest_predictions(db)
    classifier = store.active_experiment(db, "mlp")
    autoencoder = store.active_experiment(db, "autoencoder")
    classifier_metrics = json.loads(classifier.metrics_json) if classifier else {}

    charts, movers = {}, {"down": [], "up": []}
    forecast_counts, transitions = [0, 0, 0], None
    if prepared is not None:
        years, counts = queries.class_counts_by_year(prepared)
        charts["classes"] = plots.to_svg(plots.class_distribution(
            years, counts, CLASS_NAMES, "Головные организации по классам результативности"))
        current = queries.current_classes(prepared)
        if predictions:
            transitions = np.zeros((3, 3), dtype=int)
            names = {u.id: u.short_name for u in db.scalars(
                select(University).where(University.id.in_(list(predictions))))}
            for inst_id, item in predictions.items():
                forecast_counts[item["predicted_class"]] += 1
                now_class = current.get(inst_id)
                if now_class is None:
                    continue
                transitions[now_class, item["predicted_class"]] += 1
                if item["predicted_class"] != now_class:
                    key = "down" if item["predicted_class"] < now_class else "up"
                    movers[key].append({"id": inst_id, "name": names.get(inst_id, str(inst_id)),
                                        "now": now_class, "next": item["predicted_class"],
                                        "confidence": max(item["probabilities"])})
            for key in movers:
                movers[key] = sorted(movers[key], key=lambda row: row["confidence"], reverse=True)[:8]
            target_year = next(iter(predictions.values()))["target_year"]
            charts["transitions"] = plots.to_svg(plots.confusion_matrix(
                transitions, CLASS_NAMES, f"Ожидаемые переходы между классами к {target_year} году",
                xlabel=f"Прогноз на {target_year} год", ylabel="Класс в текущем году"))

    home = None
    home_id = get_setting(db, "home_university_id")
    if home_id.isdigit():
        home_university = db.get(University, int(home_id))
        if home_university is not None:
            home = {"university": home_university, "prediction": predictions.get(home_university.id),
                    "current": queries.current_classes(prepared).get(home_university.id) if prepared else None}

    recent = db.scalars(select(AuditLog).order_by(AuditLog.created_at.desc()).limit(8)).all() \
        if security.has_role(*security.ROLE_ADMIN) else []
    target_year = next(iter(predictions.values()))["target_year"] if predictions else None
    return render_template("dashboard.html", crumbs=[], summary=summary, charts=charts, movers=movers,
                           forecast_counts=forecast_counts, classifier=classifier, autoencoder=autoencoder,
                           classifier_metrics=classifier_metrics, home=home, recent=recent,
                           target_year=target_year, has_predictions=bool(predictions),
                           anomalies=sum(1 for item in predictions.values()
                                         if item["anomaly_score"] is not None and autoencoder is not None
                                         and item["anomaly_score"] > json.loads(autoencoder.metrics_json).get("threshold", 9e9)))


# --------------------------------------------------------------------------
# Реестр организаций
# --------------------------------------------------------------------------
@bp.route("/universities")
@security.login_required
def universities():
    """Список организаций с поиском, фильтрами и постраничным выводом."""
    db = get_session()
    args = request.args
    query = select(University).join(Region, Region.id == University.region_id)
    search = args.get("q", "").strip()
    if search:
        query = query.where(queries.name_filter(search))
    if args.get("region", "").isdigit():
        query = query.where(University.region_id == int(args["region"]))
    if args.get("district", "").isdigit():
        query = query.where(Region.district_id == int(args["district"]))
    if args.get("founder", "").isdigit():
        query = query.where(University.founder_group_id == int(args["founder"]))
    if args.get("kind") in ("head", "branch"):
        query = query.where(University.is_branch.is_(args["kind"] == "branch"))

    predictions = queries.latest_predictions(db)
    predicted = args.get("predicted", "")
    if predicted in ("0", "1", "2"):
        ids = [inst_id for inst_id, item in predictions.items() if item["predicted_class"] == int(predicted)]
        query = query.where(University.id.in_(ids))

    total = db.scalar(select(func.count()).select_from(query.subquery()))
    size = _page_size(db)
    page = max(1, int(args.get("page", "1")) if args.get("page", "1").isdigit() else 1)
    pages = max(1, (total + size - 1) // size)
    page = min(page, pages)
    rows = db.scalars(query.order_by(University.is_branch, University.short_name)
                      .offset((page - 1) * size).limit(size)).all()

    prepared = _prepared_or_none(db)
    current = queries.current_classes(prepared) if prepared else {}
    filters = {key: value for key, value in args.items() if key != "page" and value}
    return render_template(
        "universities.html", crumbs=[("Организации", None)], rows=rows, total=total, page=page, pages=pages,
        filters=filters, predictions=predictions, current=current,
        regions=db.scalars(select(Region).order_by(Region.name)).all(),
        districts=db.scalars(select(FederalDistrict).order_by(FederalDistrict.name)).all(),
        founders=db.scalars(select(FounderGroup).order_by(FounderGroup.name)).all())


# --------------------------------------------------------------------------
# Карточка организации
# --------------------------------------------------------------------------
def explain_prediction(db, prepared, university_id: int) -> dict:
    """Объяснение прогноза и оценка нетипичности для одной организации (по активным моделям)."""
    result = {"contributions": [], "errors": []}
    if prepared is None:
        return result
    future = prepared.parts["future"]
    mask = (future.meta["inst_id"] == university_id).to_numpy()
    if not mask.any():
        return result
    row = future.features[mask]
    titles = queries.indicator_titles(db)
    _, classifier = store.active_model(db, "mlp")
    if classifier is not None:
        x = classifier.preprocessor.transform(row)
        names = classifier.preprocessor.feature_names
        items = explain.local_contributions(lambda data: trainer.predict_proba(classifier.model, data),
                                            x[0], names, top=8)
        scale = max((abs(item["contribution"]) for item in items), default=1.0) or 1.0
        for item in items:
            item["title"] = explain.feature_title(item["feature"], titles)
            item["width"] = round(abs(item["contribution"]) / scale * 50, 1)
        result["contributions"] = items
    _, autoencoder = store.active_model(db, "autoencoder")
    if autoencoder is not None:
        items = pipeline.top_error_features(autoencoder, row, top=5)
        for item in items:
            item["title"] = explain.feature_title(item["feature"], titles)
        result["errors"] = items
        result["threshold"] = autoencoder.meta.get("threshold")
    return result


@bp.route("/universities/<int:university_id>", methods=["GET", "POST"])
@security.login_required
def university(university_id: int):
    """Карточка организации: паспорт, показатели, прогноз, объяснение, заметки."""
    db = get_session()
    item = db.get(University, university_id)
    if item is None:
        abort(404)

    # Добавление и удаление заметок доступно аналитику и администратору
    if request.method == "POST":
        if not security.has_role(*security.ROLE_ANALYST):
            abort(403)
        if request.form.get("action") == "delete_note":
            note = db.get(AnalystNote, int(request.form.get("note_id", "0") or 0))
            if note is not None and note.university_id == university_id:
                db.delete(note)
                db.commit()
                security.audit("note_delete", f"Удалена заметка к организации {item.short_name}")
        else:
            text = request.form.get("text", "").strip()
            if len(text) < 3:
                flash("Текст заметки слишком короткий.", "error")
            else:
                db.add(AnalystNote(university_id=university_id, user_id=security.current_user().id,
                                   text=text[:4000]))
                db.commit()
                security.audit("note_add", f"Добавлена заметка к организации {item.short_name}")
                flash("Заметка сохранена.", "success")
        return redirect(url_for("main.university", university_id=university_id) + "#notes")

    values = queries.university_values(db, university_id)
    medians = queries.university_medians(db, item)
    table = queries.main_indicator_table(values, medians, item.is_branch) if not values.empty else None

    chart = None
    if table and table["rows"]:
        series = [{"title": row["code"].upper().replace("E", "E.") + " " + row["title"].split(" (")[0][:22],
                   "years": [cell["year"] for cell in row["cells"]],
                   "values": [cell["value"] for cell in row["cells"]],
                   "medians": [cell["median"] for cell in row["cells"]]} for row in table["rows"]]
        chart = plots.to_svg(plots.indicator_dynamics(series))

    # Дополнительные характеристики: последний год и изменение к предыдущему
    extra_sections = []
    if not values.empty:
        last_year = int(values.index.max())
        previous_year = last_year - 1 if (last_year - 1) in values.index else None
        sections = {}
        for indicator in queries.indicators_by_column(db).values():
            if indicator.kind != "extra" or indicator.column not in values.columns:
                continue
            value = values.loc[last_year, indicator.column]
            previous = values.loc[previous_year, indicator.column] if previous_year else None
            sections.setdefault(indicator.section, []).append({
                "code": indicator.code, "name": indicator.name, "unit": indicator.unit,
                "value": None if value != value else float(value),
                "previous": None if previous is None or previous != previous else float(previous)})
        for name, rows in sections.items():
            rows.sort(key=lambda row: [int(part) for part in row["code"].split(".")])
            extra_sections.append({"name": name, "rows": rows})
        extra_year = last_year
    else:
        extra_year = None

    prepared = _prepared_or_none(db)
    prediction = queries.latest_predictions(db).get(university_id)
    explanation = explain_prediction(db, prepared, university_id) if prediction else {"contributions": [], "errors": []}
    notes = db.scalars(select(AnalystNote).where(AnalystNote.university_id == university_id)
                       .order_by(AnalystNote.created_at.desc())).all()
    history = db.scalars(select(Prediction).where(Prediction.university_id == university_id)
                         .order_by(Prediction.target_year.desc())).all()
    head = db.get(University, item.head_id) if item.head_id else None
    branches = db.scalars(select(University).where(University.head_id == university_id)
                          .order_by(University.short_name)).all()
    return render_template(
        "university.html", crumbs=[("Организации", url_for("main.universities")), (item.short_name[:60], None)],
        item=item, table=table, chart=chart, extra_sections=extra_sections, extra_year=extra_year,
        prediction=prediction, explanation=explanation, notes=notes, history=history, head=head,
        branches=branches)


# --------------------------------------------------------------------------
# Сравнение организаций
# --------------------------------------------------------------------------
@bp.route("/compare")
@security.login_required
def compare():
    """Сравнение до четырёх организаций по основным показателям последнего года."""
    db = get_session()
    # Каждая организация участвует в сравнении один раз: повторный код из адресной строки отбрасывается
    ids = list(dict.fromkeys(int(value) for value in request.args.getlist("id") if value.isdigit()))[:4]
    search = request.args.get("q", "").strip()
    found = []
    if search:
        found = db.scalars(select(University).where(queries.name_filter(search))
                           .order_by(University.is_branch, University.short_name).limit(15)).all()

    selected, rows, chart = [], [], None
    if ids:
        selected = [u for u in (db.get(University, i) for i in ids) if u is not None]
        tables = {}
        for item in selected:
            values = queries.university_values(db, item.id)
            if not values.empty:
                tables[item.id] = queries.main_indicator_table(values, queries.university_medians(db, item),
                                                               item.is_branch)
        codes = [code for code in target_indicators(False)
                 if any(code in [row["code"] for row in t["rows"]] for t in tables.values())]
        # Подписи рядов на графике должны различаться: у разных организаций бывают одинаковые названия
        names = [item.short_name for item in selected]
        labels = {item.id: item.short_name if names.count(item.short_name) == 1
                  else f"{item.short_name} (код {item.id})" for item in selected}
        ratios = {labels[item.id]: [] for item in selected if item.id in tables}
        for code in codes:
            line = {"code": code, "title": MAIN_INDICATOR_TITLES[code], "cells": []}
            for item in selected:
                cell = None
                table = tables.get(item.id)
                if table:
                    match = next((row for row in table["rows"] if row["code"] == code), None)
                    cell = match["cells"][-1] if match and match["cells"] else None
                line["cells"].append(cell)
                if item.id in tables:
                    ratio = cell["value"] / cell["median"] if cell and cell["value"] is not None \
                        and cell["median"] else 0.0
                    ratios[labels[item.id]].append(min(ratio, 4.0))
            rows.append(line)
        if codes and ratios:
            chart = plots.to_svg(plots.grouped_bars(
                [code.upper().replace("E", "E.") for code in codes], ratios,
                "Отношение показателей к медиане по России (1,0 — уровень медианы)",
                "раз к медиане", reference=1.0))
    return render_template("compare.html", crumbs=[("Сравнение", None)], selected=selected, rows=rows,
                           chart=chart, found=found, search=search, ids=ids)
