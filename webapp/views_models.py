# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Раздел «Модели»: запуск обучения, наблюдение за ходом обучения, сравнение
экспериментов, выбор рабочей модели, загрузка модели из файла и выгрузка.
"""

import json

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, send_file, url_for
from sqlalchemy import select

from vuzmon import config
from vuzmon.db.models import Experiment, Prediction
from vuzmon.db.session import get_session
from vuzmon.ml import explain, plots, registry
from vuzmon.ml.dataset import CLASS_NAMES
from vuzmon.ml.pipeline import SEARCH_SPACE
from vuzmon.ml.trainer import TrainConfig
from webapp import security
from webapp.services import bootstrap, jobs, queries, store

bp = Blueprint("models", __name__, url_prefix="/models")

STATUS_TITLES = {"new": "в очереди", "running": "обучается", "done": "готова", "failed": "ошибка"}


def _experiment_or_404(db, experiment_id: int) -> Experiment:
    experiment = db.get(Experiment, experiment_id)
    if experiment is None:
        abort(404)
    return experiment


def _history(experiment: Experiment) -> list:
    return [{"epoch": e.epoch, "train_loss": e.train_loss, "val_loss": e.val_loss,
             "train_metric": e.train_metric, "val_metric": e.val_metric} for e in experiment.epochs]


def _number(form, name: str, default, cast=float, low=None, high=None):
    """Читает число из формы с проверкой диапазона; при ошибке возвращает значение по умолчанию."""
    try:
        value = cast(str(form.get(name, default)).replace(",", "."))
    except (TypeError, ValueError):
        return default
    if low is not None:
        value = max(low, value)
    if high is not None:
        value = min(high, value)
    return value


def _parse_hidden(text: str, default=(128, 64)) -> list:
    """Размеры скрытых слоёв из строки вида '256, 128, 64'."""
    try:
        sizes = [int(part) for part in text.replace(";", ",").split(",") if part.strip()]
    except ValueError:
        return list(default)
    sizes = [min(max(size, 2), 2048) for size in sizes][:6]
    return sizes or list(default)


@bp.route("/")
@security.roles_required(*security.ROLE_ANALYST)
def index():
    """Перечень экспериментов и сравнение качества моделей."""
    db = get_session()
    experiments = db.scalars(select(Experiment).order_by(Experiment.id.desc())).all()
    rows, names, accuracy, f1 = [], [], [], []
    for experiment in experiments:
        metrics = json.loads(experiment.metrics_json or "{}")
        test = metrics.get("test") or {}
        rows.append({"experiment": experiment, "test": test, "val": metrics.get("val") or {},
                     "metrics": metrics})
        if test and experiment.status == "done":
            names.append(f"№{experiment.id}")
            accuracy.append(test["accuracy"])
            f1.append(test["macro_f1"])
    chart = None
    if names:
        order = list(reversed(range(len(names))))[-10:]
        chart = plots.to_svg(plots.grouped_bars(
            [names[i] for i in order],
            {"Доля верных ответов": [accuracy[i] for i in order], "Macro-F1": [f1[i] for i in order]},
            "Качество моделей на тестовом годе"))
    return render_template("models.html", crumbs=[("Модели", None)], rows=rows, chart=chart,
                           model_types=jobs.MODEL_TYPES, status_titles=STATUS_TITLES)


@bp.route("/new", methods=["GET", "POST"])
@security.roles_required(*security.ROLE_ANALYST)
def new():
    """Форма запуска обучения модели."""
    db = get_session()
    dataset = store.active_dataset(db)
    defaults = TrainConfig()
    if request.method == "POST":
        if dataset is None or not store.dataset_is_complete(dataset.location):
            flash("Нет рабочего набора данных. Загрузите данные в разделе «Данные».", "error")
            return redirect(url_for("models.new"))
        form = request.form
        model_type = form.get("model_type", "mlp")
        if model_type not in jobs.MODEL_TYPES:
            abort(400, description="Неизвестный вид модели.")
        params = {
            "hidden": _parse_hidden(form.get("hidden", "")),
            "dropout": _number(form, "dropout", defaults.dropout, float, 0.0, 0.9),
            "batch_norm": bool(form.get("batch_norm")),
            "activation": form.get("activation") if form.get("activation") in ("relu", "gelu", "elu", "tanh") else "relu",
            "optimizer": form.get("optimizer") if form.get("optimizer") in ("sgd", "adam", "adamw") else "adamw",
            "learning_rate": _number(form, "learning_rate", defaults.learning_rate, float, 1e-6, 1.0),
            "weight_decay": _number(form, "weight_decay", defaults.weight_decay, float, 0.0, 1.0),
            "batch_size": _number(form, "batch_size", defaults.batch_size, int, 8, 1024),
            "epochs": _number(form, "epochs", defaults.epochs, int, 1, 1000),
            "patience": _number(form, "patience", defaults.patience, int, 0, 200),
            "label_smoothing": _number(form, "label_smoothing", 0.0, float, 0.0, 0.3),
            "class_weighted": bool(form.get("class_weighted")),
            "aux_weight": _number(form, "aux_weight", defaults.aux_weight, float, 0.0, 10.0),
            "structure_weight": _number(form, "structure_weight", defaults.structure_weight, float, 0.0, 1.0),
            "ensemble_size": _number(form, "ensemble_size", 1, int, 1, 15),
            "scheduler": form.get("scheduler") if form.get("scheduler") in ("none", "plateau", "cosine") else "plateau",
            "seed": _number(form, "seed", 42, int, 0, 10 ** 6),
        }
        if model_type == "autoencoder":
            params.update(latent_dim=_number(form, "latent_dim", 8, int, 2, 64),
                          n_clusters=_number(form, "n_clusters", 5, int, 2, 12),
                          anomaly_quantile=_number(form, "anomaly_quantile", 0.95, float, 0.5, 0.999))
        if model_type == "search":
            params["n_trials"] = _number(form, "n_trials", 15, int, 2, 60)
        name = form.get("name", "").strip() or jobs.MODEL_TYPES[model_type]
        experiment = Experiment(name=name[:128], model_type=model_type,
                                params_json=json.dumps(params, ensure_ascii=False), dataset_id=dataset.id,
                                created_by=security.current_user().id, status="new")
        db.add(experiment)
        db.commit()
        jobs.start_training(experiment.id)
        security.audit("train_start", f"Запущено обучение: эксперимент №{experiment.id} «{experiment.name}»")
        return redirect(url_for("models.detail", experiment_id=experiment.id))
    return render_template("model_new.html", crumbs=[("Модели", url_for("models.index")), ("Новое обучение", None)],
                           defaults=defaults, model_types=jobs.MODEL_TYPES, dataset=dataset,
                           search_space=SEARCH_SPACE)


@bp.route("/<int:experiment_id>")
@security.roles_required(*security.ROLE_ANALYST)
def detail(experiment_id: int):
    """Карточка эксперимента: параметры, кривые обучения, метрики, матрица ошибок."""
    db = get_session()
    experiment = _experiment_or_404(db, experiment_id)
    metrics = json.loads(experiment.metrics_json or "{}")
    params = json.loads(experiment.params_json or "{}")
    history = _history(experiment)
    charts = {}
    if history:
        charts["curves"] = plots.to_svg(plots.learning_curves(history, metrics.get("best_epoch")))
    for part, title in (("val", "Матрица ошибок: валидация"), ("test", "Матрица ошибок: тест")):
        if metrics.get(part, {}).get("confusion_matrix"):
            charts[part] = plots.to_svg(plots.confusion_matrix(metrics[part]["confusion_matrix"],
                                                               CLASS_NAMES, title))
    if metrics.get("importance"):
        titles = queries.indicator_titles(db)
        top = metrics["importance"][:12]
        charts["importance"] = plots.to_svg(plots.horizontal_bars(
            [explain.feature_title(row["feature"], titles) for row in top],
            [row["importance"] for row in top], "Важность признаков (метод перестановок)",
            "снижение macro-F1 при перемешивании признака", errors=[row["std"] for row in top]))
    return render_template(
        "model_detail.html", crumbs=[("Модели", url_for("models.index")), (f"Эксперимент №{experiment.id}", None)],
        experiment=experiment, metrics=metrics, params=params, charts=charts, epochs=len(history),
        model_types=jobs.MODEL_TYPES, status_titles=STATUS_TITLES,
        can_activate=experiment.status == "done" and experiment.model_type in ("mlp", "search", "autoencoder"))


@bp.route("/<int:experiment_id>/status.json")
@security.roles_required(*security.ROLE_ANALYST)
def status(experiment_id: int):
    """Состояние обучения для опроса со страницы эксперимента."""
    db = get_session()
    experiment = _experiment_or_404(db, experiment_id)
    history = _history(experiment)
    params = json.loads(experiment.params_json or "{}")
    total = max(int(params.get("epochs", 1)), 1)
    chart = plots.to_svg(plots.learning_curves(history)) if len(history) >= 2 else ""
    return jsonify({"status": experiment.status, "epochs": len(history), "chart": chart,
                    "message": experiment.message or f"Эпоха {len(history)} из {total}",
                    "progress": min(len(history) / total, 0.99) if experiment.status == "running" else 1.0})


@bp.route("/<int:experiment_id>/activate", methods=["POST"])
@security.roles_required(*security.ROLE_ANALYST)
def activate(experiment_id: int):
    """Выбор модели в качестве рабочей."""
    db = get_session()
    experiment = _experiment_or_404(db, experiment_id)
    if experiment.status != "done" or experiment.model_type not in ("mlp", "search", "autoencoder"):
        flash("Рабочей можно сделать только готовую нейросетевую модель.", "error")
    else:
        store.activate(db, experiment)
        security.audit("model_activate", f"Модель эксперимента №{experiment.id} сделана рабочей")
        flash("Модель сделана рабочей. Чтобы обновить прогноз, перейдите в раздел «Прогноз».", "success")
    return redirect(url_for("models.detail", experiment_id=experiment.id))


@bp.route("/<int:experiment_id>/download")
@security.roles_required(*security.ROLE_ANALYST)
def download(experiment_id: int):
    """Выгрузка файла обученной модели."""
    db = get_session()
    experiment = _experiment_or_404(db, experiment_id)
    path = store.resolve(experiment.model_file) if experiment.model_file else None
    if path is None or not path.exists():
        abort(404)
    security.audit("model_download", f"Выгружен файл модели эксперимента №{experiment.id}")
    return send_file(path, as_attachment=True, download_name=path.name)


@bp.route("/<int:experiment_id>/delete", methods=["POST"])
@security.roles_required(*security.ROLE_ANALYST)
def delete(experiment_id: int):
    """Удаление эксперимента вместе с файлом модели и построенными по ней прогнозами."""
    db = get_session()
    experiment = _experiment_or_404(db, experiment_id)
    if experiment.status == "running":
        flash("Нельзя удалить эксперимент во время обучения.", "error")
        return redirect(url_for("models.detail", experiment_id=experiment.id))
    if experiment.is_active:
        flash("Нельзя удалить рабочую модель. Сначала выберите другую.", "error")
        return redirect(url_for("models.detail", experiment_id=experiment.id))
    for prediction in db.scalars(select(Prediction).where(Prediction.experiment_id == experiment.id)):
        db.delete(prediction)
    path = store.resolve(experiment.model_file) if experiment.model_file else None
    # Удаляем только файлы из каталога моделей системы
    if path is not None and path.exists() and path.parent.resolve() == config.MODELS_DIR.resolve():
        path.unlink()
    name = experiment.name
    db.delete(experiment)
    db.commit()
    security.audit("model_delete", f"Удалён эксперимент «{name}»")
    flash("Эксперимент удалён.", "success")
    return redirect(url_for("models.index"))


@bp.route("/upload", methods=["POST"])
@security.roles_required(*security.ROLE_ANALYST)
def upload():
    """Загрузка обученной модели из файла (*.pt) и её регистрация в системе."""
    db = get_session()
    uploaded = request.files.get("file")
    if uploaded is None or not uploaded.filename.lower().endswith(".pt"):
        flash("Выберите файл модели с расширением .pt.", "error")
        return redirect(url_for("models.index"))
    path = registry.next_model_path(config.MODELS_DIR, ".pt")
    uploaded.save(path)
    try:
        experiment = bootstrap.register_model_file(
            db, path, name=request.form.get("name", "").strip()[:128] or f"Загружена из файла {uploaded.filename}",
            user_id=security.current_user().id)
    except Exception as error:   # файл не является моделью системы или повреждён
        path.unlink(missing_ok=True)
        flash(f"Файл не удалось загрузить как модель: {error}", "error")
        return redirect(url_for("models.index"))
    security.audit("model_upload", f"Загружена модель из файла: эксперимент №{experiment.id}")
    flash("Модель загружена из файла и готова к применению.", "success")
    return redirect(url_for("models.detail", experiment_id=experiment.id))
