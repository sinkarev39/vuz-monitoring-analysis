# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Разделы «Прогноз» и «Нетипичные профили»: применение обученных моделей.
"""

import numpy as np
from flask import Blueprint, abort, flash, redirect, render_template, request, url_for
from sqlalchemy import func, select

from vuzmon.config import MAIN_INDICATOR_TITLES
from vuzmon.db.models import FederalDistrict, Region, University
from vuzmon.db.session import get_session, get_setting
from vuzmon.ml import explain, pipeline, plots, trainer
from vuzmon.ml.dataset import CLASS_NAMES, HEAD_TARGET_INDICATORS, apply_scenario
from webapp import security
from webapp.services import jobs, queries, store

bp = Blueprint("forecast", __name__)


def _prepared(db):
    dataset = store.active_dataset(db)
    if dataset is None or not store.dataset_is_complete(dataset.location):
        return None
    return store.prepared(dataset.location)


# --------------------------------------------------------------------------
# Прогноз
# --------------------------------------------------------------------------
@bp.route("/forecast")
@security.login_required
def index():
    """Таблица прогнозов на следующий год с фильтрами."""
    db = get_session()
    predictions = queries.latest_predictions(db)
    prepared = _prepared(db)
    current = queries.current_classes(prepared) if prepared else {}
    experiment = store.active_experiment(db, "mlp")
    args = request.args

    query = select(University).join(Region, Region.id == University.region_id) \
        .where(University.id.in_(list(predictions) or [-1]))
    search = args.get("q", "").strip()
    if search:
        query = query.where(func.lower(University.name).like(f"%{search.lower()}%"))
    if args.get("district", "").isdigit():
        query = query.where(Region.district_id == int(args["district"]))
    rows = []
    for item in db.scalars(query):
        prediction = predictions[item.id]
        now_class = current.get(item.id)
        change = 0 if now_class is None else prediction["predicted_class"] - now_class
        if args.get("predicted") in ("0", "1", "2") and prediction["predicted_class"] != int(args["predicted"]):
            continue
        if args.get("change") == "down" and change >= 0:
            continue
        if args.get("change") == "up" and change <= 0:
            continue
        if args.get("change") == "same" and change != 0:
            continue
        rows.append({"item": item, "prediction": prediction, "current": now_class, "change": change})

    # Сортировка: по вероятности зоны риска, по уверенности или по названию
    sort = args.get("sort", "risk")
    if sort == "name":
        rows.sort(key=lambda row: row["item"].short_name)
    elif sort == "confidence":
        rows.sort(key=lambda row: max(row["prediction"]["probabilities"]))
    else:
        rows.sort(key=lambda row: row["prediction"]["probabilities"][0], reverse=True)

    size = max(5, min(200, int(get_setting(db, "records_per_page", "25") or 25)))
    total = len(rows)
    pages = max(1, (total + size - 1) // size)
    page = min(max(1, int(args.get("page", "1")) if args.get("page", "1").isdigit() else 1), pages)
    counts = [sum(1 for p in predictions.values() if p["predicted_class"] == label) for label in range(3)]
    first = next(iter(predictions.values()), None)
    return render_template(
        "forecast.html", crumbs=[("Прогноз", None)], rows=rows[(page - 1) * size:page * size], total=total,
        page=page, pages=pages, filters={k: v for k, v in args.items() if k != "page" and v},
        counts=counts, experiment=experiment, base_year=first["base_year"] if first else None,
        target_year=first["target_year"] if first else None, built_at=first["created_at"] if first else None,
        districts=db.scalars(select(FederalDistrict).order_by(FederalDistrict.name)).all())


@bp.route("/forecast/build", methods=["POST"])
@security.roles_required(*security.ROLE_ANALYST)
def build():
    """Построение прогноза рабочей моделью по данным последнего года мониторинга."""
    db = get_session()
    try:
        outcome = jobs.build_forecast(db, security.current_user().id)
    except RuntimeError as error:
        flash(str(error), "error")
        return redirect(url_for("forecast.index"))
    security.audit("forecast_build", f"Построен прогноз на {outcome['base_year'] + 1} год: "
                                     f"организаций {outcome['count']}, модель №{outcome['experiment_id']}")
    flash(f"Прогноз построен для {outcome['count']} организаций по данным {outcome['base_year']} года.", "success")
    return redirect(url_for("forecast.index"))


@bp.route("/forecast/what-if/<int:university_id>", methods=["GET", "POST"])
@security.login_required
def what_if(university_id: int):
    """
    Расчёт «что будет, если…»: пользователь задаёт значения основных
    показателей, система пересчитывает признаки и показывает новый прогноз.
    """
    db = get_session()
    item = db.get(University, university_id)
    if item is None:
        abort(404)
    prepared = _prepared(db)
    _, classifier = store.active_model(db, "mlp")
    if prepared is None or classifier is None:
        flash("Для расчёта нужны рабочий набор данных и рабочая модель прогноза.", "error")
        return redirect(url_for("main.university", university_id=university_id))
    future = prepared.parts["future"]
    mask = (future.meta["inst_id"] == university_id).to_numpy()
    if not mask.any():
        flash("Для этой организации прогноз не рассчитывается (нет данных последнего года).", "warning")
        return redirect(url_for("main.university", university_id=university_id))
    features = future.features[mask]
    year = int(future.meta.loc[mask, "year"].iloc[0])

    values = queries.university_values(db, university_id)
    fields, current, medians, previous = [], {}, {}, {}
    for code in HEAD_TARGET_INDICATORS:
        current[code] = float(values.loc[year, code]) if code in values.columns and values.loc[year, code] == values.loc[year, code] else 0.0
        median = values.loc[year, f"med_{code}"] if f"med_{code}" in values.columns else None
        medians[code] = float(median) if median is not None and median == median else None
        before = values.loc[year - 1, code] if (year - 1) in values.index and code in values.columns else None
        previous[code] = float(before) if before is not None and before == before else None

    scenario = dict(current)
    if request.method == "POST":
        for code in HEAD_TARGET_INDICATORS:
            try:
                scenario[code] = max(0.0, float(request.form.get(code, "").replace(",", ".").replace(" ", "")))
            except ValueError:
                flash(f"Значение показателя {code.upper()} не распознано, оставлено исходное.", "warning")

    def predict(frame):
        return trainer.predict_proba(classifier.model, classifier.preprocessor.transform(frame))[0]

    base_probabilities = predict(features)
    new_features = apply_scenario(features, scenario, medians, previous)
    new_probabilities = predict(new_features)
    for code in HEAD_TARGET_INDICATORS:
        fields.append({"code": code, "title": MAIN_INDICATOR_TITLES[code], "current": current[code],
                       "scenario": scenario[code], "median": medians[code],
                       "passed": medians[code] is not None and scenario[code] >= medians[code] and scenario[code] > 0})
    if request.method == "POST":
        security.audit("what_if", f"Расчёт «что будет, если» для организации {item.short_name}")
    return render_template(
        "what_if.html", crumbs=[("Прогноз", url_for("forecast.index")), (item.short_name[:50], url_for("main.university", university_id=item.id)), ("Что будет, если…", None)],
        item=item, fields=fields, year=year, base=[float(p) for p in base_probabilities],
        new=[float(p) for p in new_probabilities], changed=request.method == "POST",
        base_class=int(np.argmax(base_probabilities)), new_class=int(np.argmax(new_probabilities)))


# --------------------------------------------------------------------------
# Нетипичные профили и типологические группы
# --------------------------------------------------------------------------
@bp.route("/profiles")
@security.login_required
def profiles():
    """Результаты автокодировщика: карта организаций, группы, нетипичные записи."""
    db = get_session()
    prepared = _prepared(db)
    experiment, autoencoder = store.active_model(db, "autoencoder")
    if prepared is None or autoencoder is None:
        return render_template("profiles.html", crumbs=[("Нетипичные профили", None)], ready=False)

    future = prepared.parts["future"]
    table = pipeline.profile(autoencoder, future)
    latent = table[[c for c in table.columns if c.startswith("z")]].to_numpy()
    points = plots.pca_2d(latent)
    threshold = float(autoencoder.meta.get("threshold", np.inf))
    n_clusters = len(autoencoder.meta.get("centers", [])) or 1
    clusters = table["cluster"].to_numpy() if "cluster" in table else np.zeros(len(table), dtype=int)
    current = future.meta["current_class"].to_numpy()

    charts = {
        "map": plots.to_svg(plots.scatter_map(
            points[:, 0], points[:, 1], clusters, [f"Группа {i + 1}" for i in range(n_clusters)],
            highlight=table["is_anomaly"].to_numpy(), title="Карта организаций в пространстве скрытых признаков",
            xlabel="Главная компонента 1", ylabel="Главная компонента 2")),
        "errors": plots.to_svg(plots.histogram(
            table["anomaly_score"], "Распределение ошибки восстановления", "Ошибка восстановления (MSE)",
            threshold, "порог нетипичности")),
    }

    # Описание групп: размер, состав по классам, отличительные признаки
    titles = queries.indicator_titles(db)
    x = autoencoder.preprocessor.transform(future.features)
    names = autoencoder.preprocessor.feature_names
    groups = []
    for label in range(n_clusters):
        members = clusters == label
        if not members.any():
            continue
        means = x[members].mean(axis=0)
        top = np.argsort(-np.abs(means))[:4]
        groups.append({
            "number": label + 1, "size": int(members.sum()),
            "classes": [int(((current == c) & members).sum()) for c in range(3)],
            "anomalies": int((table["is_anomaly"].to_numpy() & members).sum()),
            "features": [{"title": explain.feature_title(names[i], titles), "value": float(means[i])} for i in top]})

    order = np.argsort(-table["anomaly_score"].to_numpy())[:25]
    universities = {u.id: u for u in db.scalars(select(University).where(
        University.id.in_([int(v) for v in table["inst_id"].iloc[order]])))}
    anomalies = []
    for position in order:
        row = table.iloc[position]
        reasons = pipeline.top_error_features(autoencoder, future.features.iloc[[position]], top=3)
        anomalies.append({"item": universities.get(int(row["inst_id"])), "name": row["name"],
                          "score": float(row["anomaly_score"]), "flag": bool(row["is_anomaly"]),
                          "cluster": int(row["cluster"]) + 1 if "cluster" in table else None,
                          "reasons": [explain.feature_title(r["feature"], titles) for r in reasons]})
    return render_template("profiles.html", crumbs=[("Нетипичные профили", None)], ready=True, charts=charts,
                           groups=groups, anomalies=anomalies, threshold=threshold, experiment=experiment,
                           total=len(table), flagged=int(table["is_anomaly"].sum()),
                           year=int(future.meta["year"].iloc[0]) if len(future) else None)
