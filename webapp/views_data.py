# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Раздел «Данные»: сбор с сайта мониторинга, загрузка набора данных из
локальных файлов, просмотр, выгрузка и выбор рабочего набора.
"""

import io
import shutil
import zipfile
from datetime import datetime

from flask import Blueprint, abort, flash, jsonify, redirect, render_template, request, send_file, url_for
from sqlalchemy import select

from vuzmon import config
from vuzmon.collect import validation
from vuzmon.collect.crawler import DATASET_FILES
from vuzmon.db.loader import load_monitoring_frames
from vuzmon.db.models import Dataset, Experiment
from vuzmon.db.session import get_session
from webapp import security
from webapp.services import jobs, store

bp = Blueprint("data", __name__, url_prefix="/data")

TABLE_TITLES = {"institutions": "Паспорта организаций", "main": "Основные показатели",
                "medians": "Медианные значения", "extra": "Дополнительные характеристики",
                "dictionary": "Справочник показателей"}


def _dataset_or_404(db, dataset_id: int) -> Dataset:
    dataset = db.get(Dataset, dataset_id)
    if dataset is None:
        abort(404)
    return dataset


@bp.route("/")
@security.roles_required(*security.ROLE_ANALYST)
def index():
    """Перечень наборов данных и формы получения новых данных."""
    db = get_session()
    datasets = db.scalars(select(Dataset).order_by(Dataset.id.desc())).all()
    active = store.active_dataset(db)
    job = jobs.latest_job("collect")
    return render_template("data.html", crumbs=[("Данные", None)], datasets=datasets, active=active, job=job,
                           years=list(range(2021, datetime.now().year + 1)),
                           default_years=config.DEFAULT_YEARS, store=store)


@bp.route("/collect", methods=["POST"])
@security.roles_required(*security.ROLE_ANALYST)
def collect():
    """Запуск сбора данных с сайта мониторинга за выбранные годы."""
    years = sorted({int(value) for value in request.form.getlist("years") if value.isdigit()})
    if not years:
        flash("Выберите хотя бы один год мониторинга.", "error")
        return redirect(url_for("data.index"))
    if jobs.latest_job("collect").get("status") == "running":
        flash("Сбор данных уже выполняется. Дождитесь его завершения.", "warning")
        return redirect(url_for("data.index"))
    limit = request.form.get("limit", "")
    jobs.start_collect(years, security.current_user().id, load_to_db=bool(request.form.get("load_to_db")),
                       limit=int(limit) if limit.isdigit() and int(limit) > 0 else None)
    security.audit("collect_start", f"Запущен сбор данных с сайта мониторинга за годы: {years}")
    flash("Сбор данных запущен. Ход выполнения отображается на этой странице.", "info")
    return redirect(url_for("data.index"))


@bp.route("/job.json")
@security.roles_required(*security.ROLE_ANALYST)
def job_state():
    """Состояние последней задачи сбора данных (для опроса со страницы)."""
    job = jobs.latest_job("collect")
    return jsonify({"status": job.get("status", "none"), "message": job.get("message", ""),
                    "progress": job.get("progress", 0.0)})


def _read_uploaded_tables(files) -> dict:
    """
    Читает загруженные файлы: либо один ZIP-архив с пятью CSV, либо сами CSV.
    Возвращает словарь {имя таблицы: DataFrame}; при ошибке выбрасывает ValueError.
    """
    contents = {}   # имя файла -> байты
    for uploaded in files:
        name = (uploaded.filename or "").replace("\\", "/").split("/")[-1].lower()
        data = uploaded.read()
        if name.endswith(".zip"):
            try:
                with zipfile.ZipFile(io.BytesIO(data)) as archive:
                    for member in archive.namelist():
                        base = member.split("/")[-1].lower()
                        if base in DATASET_FILES.values():
                            contents[base] = archive.read(member)
            except zipfile.BadZipFile:
                raise ValueError(f"файл {name} не является ZIP-архивом или повреждён") from None
        elif name in DATASET_FILES.values():
            contents[name] = data
    return {table: validation.read_csv_table(contents, filename) for table, filename in DATASET_FILES.items()}


@bp.route("/upload", methods=["POST"])
@security.roles_required(*security.ROLE_ANALYST)
def upload():
    """Загрузка набора данных из локальных файлов (ZIP-архив или пять CSV-файлов)."""
    db = get_session()
    files = [f for f in request.files.getlist("files") if f and f.filename]
    if not files:
        flash("Выберите файлы набора данных.", "error")
        return redirect(url_for("data.index"))
    try:
        tables = _read_uploaded_tables(files)
    except ValueError as error:
        flash(f"Набор данных не загружен: {error}.", "error")
        return redirect(url_for("data.index"))
    # Содержимое проверяется до сохранения: набор с ошибками на диск и в перечень не попадает
    problems = validation.check_frames(tables)
    if problems:
        flash("Набор данных не загружен: в файлах найдены ошибки.", "error")
        for problem in problems:
            flash(f"{problem}.", "error")
        return redirect(url_for("data.index"))

    directory = store.new_dataset_directory("upload")
    for table, filename in DATASET_FILES.items():
        tables[table].to_csv(directory / filename, index=False, encoding="utf-8")
    years = sorted(int(year) for year in tables["institutions"]["year"].unique())
    name = request.form.get("name", "").strip() or f"Загрузка из файла от {datetime.now():%d.%m.%Y %H:%M}"
    dataset = Dataset(name=name[:128], source="file", location=store.relative(directory),
                      years=", ".join(map(str, years)), rows_count=len(tables["institutions"]),
                      created_by=security.current_user().id,
                      description=request.form.get("description", "").strip()[:1000])
    db.add(dataset)
    db.commit()
    security.audit("dataset_upload", f"Загружен набор данных №{dataset.id} «{dataset.name}»")
    flash(f"Набор данных «{dataset.name}» загружен: организаций и лет — {dataset.rows_count}.", "success")
    return redirect(url_for("data.detail", dataset_id=dataset.id))


@bp.route("/<int:dataset_id>")
@security.roles_required(*security.ROLE_ANALYST)
def detail(dataset_id: int):
    """Сведения о наборе данных: состав по годам, полнота, первые строки таблиц."""
    db = get_session()
    dataset = _dataset_or_404(db, dataset_id)
    if not store.dataset_is_complete(dataset.location):
        flash("Файлы набора данных не найдены на диске.", "error")
        return redirect(url_for("data.index"))
    frames = store.frames(dataset.location)
    institutions = frames["institutions"]
    by_year = institutions.groupby("year").agg(total=("inst_id", "count"), branches=("is_branch", "sum")) \
        .reset_index().to_dict("records")
    tables = []
    for name, frame in frames.items():
        numeric = frame.select_dtypes("number")
        tables.append({
            "name": name, "title": TABLE_TITLES[name], "file": DATASET_FILES[name], "rows": len(frame),
            "columns": len(frame.columns),
            "missing": float(numeric.isna().mean().mean()) if not numeric.empty else 0.0,
            "preview": frame.head(5).iloc[:, :9].astype(object).where(frame.head(5).iloc[:, :9].notna(), "—")
                            .to_dict("split")})
    return render_template("data_detail.html",
                           crumbs=[("Данные", url_for("data.index")), (f"Набор №{dataset.id}", None)],
                           dataset=dataset, by_year=by_year, tables=tables,
                           is_active=(store.active_dataset(db) or dataset).id == dataset.id)


@bp.route("/<int:dataset_id>/download")
@security.roles_required(*security.ROLE_ANALYST)
def download(dataset_id: int):
    """Выгрузка набора данных одним ZIP-архивом."""
    db = get_session()
    dataset = _dataset_or_404(db, dataset_id)
    directory = store.resolve(dataset.location)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in DATASET_FILES.values():
            if (directory / filename).exists():
                archive.write(directory / filename, filename)
    buffer.seek(0)
    security.audit("dataset_download", f"Выгружен набор данных №{dataset.id}")
    return send_file(buffer, mimetype="application/zip", as_attachment=True,
                     download_name=f"monitoring_dataset_{dataset.id}.zip")


@bp.route("/<int:dataset_id>/activate", methods=["POST"])
@security.roles_required(*security.ROLE_ANALYST)
def activate(dataset_id: int):
    """Делает набор данных рабочим и перезагружает его в базу данных."""
    db = get_session()
    dataset = _dataset_or_404(db, dataset_id)
    # Набор, по которому нельзя построить выборку, рабочим не делаем: иначе перестанут открываться страницы
    problem = store.activation_problem(dataset.location)
    if problem:
        flash(f"Набор данных №{dataset.id} нельзя сделать рабочим. {problem}", "error")
        return redirect(url_for("data.index"))
    stats = load_monitoring_frames(db, store.frames(dataset.location))
    store.set_active_dataset(db, dataset.id)
    security.audit("dataset_activate", f"Набор данных №{dataset.id} сделан рабочим: {stats}")
    flash(f"Набор данных №{dataset.id} загружен в базу: организаций — {stats['universities']}, "
          f"значений показателей — {stats['indicator_values']}.", "success")
    return redirect(url_for("data.index"))


@bp.route("/<int:dataset_id>/delete", methods=["POST"])
@security.roles_required(*security.ROLE_ANALYST)
def delete(dataset_id: int):
    """Удаление набора данных (кроме рабочего и использованного в экспериментах)."""
    db = get_session()
    dataset = _dataset_or_404(db, dataset_id)
    active = store.active_dataset(db)
    if active is not None and active.id == dataset.id:
        flash("Нельзя удалить рабочий набор данных.", "error")
    elif db.scalars(select(Experiment).where(Experiment.dataset_id == dataset.id)).first() is not None:
        flash("Набор данных использован в экспериментах и не может быть удалён.", "error")
    else:
        directory = store.resolve(dataset.location)
        # Удаляем только каталоги, созданные самой системой внутри data/datasets
        if directory.exists() and (config.DATA_DIR / "datasets").resolve() in directory.resolve().parents:
            shutil.rmtree(directory)
        name = dataset.name
        db.delete(dataset)
        db.commit()
        security.audit("dataset_delete", f"Удалён набор данных «{name}»")
        flash("Набор данных удалён.", "success")
    return redirect(url_for("data.index"))
