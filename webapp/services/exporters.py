# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль exporters — формирование документов Word (.docx) и таблиц Excel (.xlsx).

Документы собираются в памяти и отдаются пользователю как файлы для
скачивания; на диске сервера они не сохраняются.
"""

import io
from datetime import datetime

from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Cm, Pt, RGBColor
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

from vuzmon.ml import plots
from vuzmon.ml.dataset import CLASS_NAMES

SYSTEM_NAME = "Система интеллектуального анализа данных мониторинга вузов"
CLASS_FILLS = ("F4CCCC", "E3E7EC", "CFE2F3")     # заливка ячеек Excel по классам
HEADER_FILL = "1F4E79"


def _num(value, digits: int = 2) -> str:
    """Число в российской записи для документов."""
    if value is None or value != value:
        return "—"
    text = f"{value:,.{digits}f}" if isinstance(value, float) and not float(value).is_integer() else f"{int(value):,}"
    return text.replace(",", " ").replace(".", ",")


def _pct(value) -> str:
    return "—" if value is None else f"{value * 100:.1f}".replace(".", ",") + " %"


# --------------------------------------------------------------------------
# Общие элементы документов Word
# --------------------------------------------------------------------------
def _new_document(title: str, subtitle: str = "") -> Document:
    """Создаёт документ с полями по ГОСТ, шрифтом Times New Roman и заголовком."""
    document = Document()
    section = document.sections[0]
    section.left_margin, section.right_margin = Cm(3), Cm(1.5)
    section.top_margin, section.bottom_margin = Cm(2), Cm(2)
    style = document.styles["Normal"]
    style.font.name = "Times New Roman"
    style.font.size = Pt(12)
    heading = document.add_paragraph()
    heading.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = heading.add_run(title)
    run.bold = True
    run.font.size = Pt(14)
    if subtitle:
        line = document.add_paragraph(subtitle)
        line.alignment = WD_ALIGN_PARAGRAPH.CENTER
    return document


def _heading(document: Document, text: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(10)
    run = paragraph.add_run(text)
    run.bold = True
    run.font.size = Pt(13)


def _table(document: Document, header: list, rows: list, widths: list = None, font_size: int = 10):
    """Таблица с рамкой и полужирной шапкой."""
    table = document.add_table(rows=1, cols=len(header))
    table.style = "Table Grid"
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    for cell, text in zip(table.rows[0].cells, header):
        cell.text = ""
        run = cell.paragraphs[0].add_run(str(text))
        run.bold = True
        run.font.size = Pt(font_size)
    for values in rows:
        cells = table.add_row().cells
        for cell, value in zip(cells, values):
            cell.text = ""
            cell.paragraphs[0].add_run(str(value)).font.size = Pt(font_size)
    if widths:
        for row in table.rows:
            for cell, width in zip(row.cells, widths):
                cell.width = Cm(width)
    return table


def _footer_note(document: Document, user_name: str) -> None:
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(14)
    run = paragraph.add_run(f"Документ сформирован {datetime.now():%d.%m.%Y %H:%M} в информационной системе "
                            f"«{SYSTEM_NAME}». Исполнитель: {user_name}.")
    run.italic = True
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor(0x55, 0x55, 0x55)


def _to_buffer(document) -> io.BytesIO:
    buffer = io.BytesIO()
    document.save(buffer)
    buffer.seek(0)
    return buffer


# --------------------------------------------------------------------------
# Аналитическая справка по организации (Word)
# --------------------------------------------------------------------------
def university_docx(item, table: dict, prediction: dict, explanation: dict, notes: list,
                    user_name: str) -> io.BytesIO:
    document = _new_document("Аналитическая справка", item.short_name)
    _heading(document, "1. Общие сведения")
    _table(document, ["Параметр", "Значение"], [
        ("Полное наименование", item.name),
        ("Тип организации", "филиал" if item.is_branch else "головная организация"),
        ("Регион", f"{item.region.name} ({item.region.district.name})"),
        ("Ведомственная принадлежность", item.founder_group.name if item.founder_group else "—"),
        ("Отраслевая специфика", item.specifics or "—"),
        ("Адрес", item.address or "—"),
        ("Код в мониторинге", item.id),
    ], widths=[5.5, 11])

    if table and table["rows"]:
        _heading(document, "2. Основные показатели эффективности")
        years = table["years"][-5:]
        header = ["Показатель"] + [str(year) for year in years] + ["Медиана РФ"]
        rows = []
        for row in table["rows"]:
            cells = [cell for cell in row["cells"] if cell["year"] in years]
            rows.append([f"{row['code'].upper().replace('E', 'E.')} {row['title']}"]
                        + [_num(cell["value"]) for cell in cells] + [_num(cells[-1]["median"])])
        rows.append(["Выполнено показателей (не ниже медианы)"]
                    + [str(table["counts"].get(year, "—")) for year in years] + [""])
        rows.append(["Класс результативности"]
                    + [CLASS_NAMES[table["classes"][year]] if year in table["classes"] else "—" for year in years]
                    + [""])
        _table(document, header, rows, widths=[6.0] + [1.9] * len(years) + [2.0], font_size=9)

    _heading(document, "3. Прогноз")
    if prediction:
        document.add_paragraph(
            f"По данным {prediction['base_year']} года модель относит организацию в мониторинге "
            f"{prediction['target_year']} года к классу «{CLASS_NAMES[prediction['predicted_class']]}».")
        _table(document, ["Класс", "Вероятность"],
               [(name, _pct(p)) for name, p in zip(CLASS_NAMES, prediction["probabilities"])], widths=[6, 4])
        if prediction.get("anomaly_score") is not None:
            document.add_paragraph(
                f"Оценка нетипичности профиля (ошибка восстановления автокодировщика): "
                f"{_num(prediction['anomaly_score'], 3)}.")
        if explanation.get("contributions"):
            document.add_paragraph("Признаки, сильнее всего повлиявшие на прогноз:")
            for row in explanation["contributions"][:6]:
                sign = "повышает" if row["contribution"] >= 0 else "понижает"
                document.add_paragraph(
                    f"{row['title']} — {sign} вероятность предсказанного класса на "
                    f"{abs(row['contribution']) * 100:.1f} п.п.".replace(".", ","), style="List Bullet")
    else:
        document.add_paragraph("Прогноз для организации не рассчитывался.")

    _heading(document, "4. Заметки аналитиков")
    if notes:
        for note in notes:
            document.add_paragraph(f"{note.created_at:%d.%m.%Y} — {note.user.full_name or note.user.login}: {note.text}",
                                   style="List Bullet")
    else:
        document.add_paragraph("Заметок нет.")
    _footer_note(document, user_name)
    return _to_buffer(document)


# --------------------------------------------------------------------------
# Протокол обучения модели (Word)
# --------------------------------------------------------------------------
def experiment_docx(experiment, metrics: dict, params: dict, history: list, model_title: str,
                    user_name: str) -> io.BytesIO:
    document = _new_document("Протокол обучения модели", f"Эксперимент №{experiment.id}. {experiment.name}")
    _heading(document, "1. Общие сведения")
    _table(document, ["Параметр", "Значение"], [
        ("Вид модели", model_title),
        ("Набор данных", experiment.dataset.name if experiment.dataset else "—"),
        ("Начало обучения", f"{experiment.started_at:%d.%m.%Y %H:%M}"),
        ("Окончание обучения", f"{experiment.finished_at:%d.%m.%Y %H:%M}" if experiment.finished_at else "—"),
        ("Файл модели", experiment.model_file or "—"),
        ("Число обучаемых параметров", _num(metrics.get("n_parameters"))),
        ("Лучшая эпоха / выполнено эпох", f"{metrics.get('best_epoch', '—')} / {metrics.get('epochs_run', '—')}"),
    ], widths=[6, 10.5])

    _heading(document, "2. Гиперпараметры")
    _table(document, ["Параметр", "Значение"],
           [(key, value) for key, value in params.items() if key != "best"], widths=[6, 10.5])

    if metrics.get("test"):
        _heading(document, "3. Метрики качества")
        naive = metrics.get("persistence", {})
        rows = []
        for key, title, digits in (("accuracy", "Доля верных ответов", None), ("balanced_accuracy", "Сбалансированная точность", None),
                                   ("macro_f1", "Macro-F1", 3), ("macro_auc", "ROC-AUC (среднее по классам)", 3),
                                   ("log_loss", "Логарифмическая функция потерь", 3)):
            fmt = _pct if digits is None else (lambda v, d=digits: _num(float(v), d))
            rows.append([title, fmt(metrics["val"][key]), fmt(metrics["test"][key]),
                         fmt(naive["test"][key]) if naive.get("test") else "—"])
        _table(document, ["Метрика", "Валидация", "Тест", "Наивный прогноз (тест)"], rows, widths=[6.5, 3, 3, 4])

        document.add_paragraph()
        document.add_paragraph("Матрица ошибок на тестовой выборке (строки — фактический класс, столбцы — предсказанный):")
        matrix = metrics["test"]["confusion_matrix"]
        _table(document, [""] + list(CLASS_NAMES),
               [[name] + [str(v) for v in row] for name, row in zip(CLASS_NAMES, matrix)], widths=[4.5, 3.5, 3.5, 3.5])

    if len(history) >= 2:
        _heading(document, "4. Кривые обучения")
        image = io.BytesIO()
        plots.to_png(plots.learning_curves(history, metrics.get("best_epoch")), image, dpi=160)
        image.seek(0)
        document.add_picture(image, width=Cm(16))
    _footer_note(document, user_name)
    return _to_buffer(document)


# --------------------------------------------------------------------------
# Сводная записка по прогнозу (Word)
# --------------------------------------------------------------------------
def summary_docx(target_year: int, counts: list, by_district: list, decline: list, growth: list,
                 model_line: str, user_name: str) -> io.BytesIO:
    document = _new_document("Сводная аналитическая записка",
                             f"Прогноз результатов мониторинга эффективности вузов на {target_year} год")
    total = sum(counts) or 1
    _heading(document, "1. Общая картина")
    document.add_paragraph(
        f"Прогноз построен для {sum(counts)} головных образовательных организаций. {model_line}")
    _table(document, ["Класс результативности", "Организаций", "Доля"],
           [(name, count, _pct(count / total)) for name, count in zip(CLASS_NAMES, counts)], widths=[7, 4, 4])
    _heading(document, "2. Распределение по федеральным округам")
    _table(document, ["Федеральный округ"] + list(CLASS_NAMES) + ["Всего"],
           [[row["name"]] + row["counts"] + [sum(row["counts"])] for row in by_district],
           widths=[5, 3, 3.5, 3, 2.5])
    for number, title, rows in ((3, "Организации, для которых ожидается снижение класса", decline),
                                (4, "Организации, для которых ожидается повышение класса", growth)):
        _heading(document, f"{number}. {title}")
        if rows:
            _table(document, ["Организация", "Регион", "Сейчас", "Прогноз", "Уверенность"],
                   [(row["name"], row["region"], CLASS_NAMES[row["now"]], CLASS_NAMES[row["next"]],
                     _pct(row["confidence"])) for row in rows], widths=[6.5, 3.5, 2.3, 2.3, 2.2], font_size=9)
        else:
            document.add_paragraph("Таких организаций нет.")
    _footer_note(document, user_name)
    return _to_buffer(document)


# --------------------------------------------------------------------------
# Таблицы Excel
# --------------------------------------------------------------------------
def _style_header(sheet, columns: int) -> None:
    """Оформление шапки листа: заливка, белый полужирный шрифт, закрепление строки."""
    thin = Side(style="thin", color="BFC7D1")
    for column in range(1, columns + 1):
        cell = sheet.cell(row=1, column=column)
        cell.font = Font(bold=True, color="FFFFFF")
        cell.fill = PatternFill("solid", start_color=HEADER_FILL)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        cell.border = Border(left=thin, right=thin, top=thin, bottom=thin)
    sheet.freeze_panes = "A2"
    sheet.row_dimensions[1].height = 32


def forecast_xlsx(rows: list, base_year: int, target_year: int) -> io.BytesIO:
    """Книга Excel с прогнозом: лист с таблицей и лист со сводкой."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Прогноз"
    header = ["Код", "Организация", "Регион", "Федеральный округ", f"Класс в {base_year} г.",
              f"Прогноз на {target_year} г.", "P(зона риска)", "P(средний уровень)", "P(лидеры)",
              "Изменение класса", "Оценка нетипичности", "Группа"]
    sheet.append(header)
    for row in rows:
        prediction, item = row["prediction"], row["item"]
        sheet.append([item.id, item.short_name, item.region.name, item.region.district.name,
                      CLASS_NAMES[row["current"]] if row["current"] is not None else "",
                      CLASS_NAMES[prediction["predicted_class"]], *prediction["probabilities"],
                      {-2: "снижение", -1: "снижение", 0: "без изменений", 1: "рост", 2: "рост"}[row["change"]],
                      prediction["anomaly_score"],
                      prediction["cluster"] + 1 if prediction["cluster"] is not None else None])
        line = sheet.max_row
        sheet.cell(row=line, column=6).fill = PatternFill("solid", start_color=CLASS_FILLS[prediction["predicted_class"]])
        for column in (7, 8, 9):
            sheet.cell(row=line, column=column).number_format = "0.0%"
        sheet.cell(row=line, column=11).number_format = "0.000"
    _style_header(sheet, len(header))
    for index, width in enumerate([9, 60, 26, 14, 18, 18, 14, 16, 12, 16, 16, 9], start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.auto_filter.ref = sheet.dimensions

    summary = workbook.create_sheet("Сводка")
    summary.append(["Класс результативности", "Организаций по прогнозу", "Доля"])
    total = len(rows) or 1
    for label, name in enumerate(CLASS_NAMES):
        count = sum(1 for row in rows if row["prediction"]["predicted_class"] == label)
        summary.append([name, count, count / total])
        summary.cell(row=summary.max_row, column=3).number_format = "0.0%"
    summary.append([])
    summary.append([f"Сформировано {datetime.now():%d.%m.%Y %H:%M}. {SYSTEM_NAME}."])
    _style_header(summary, 3)
    for index, width in enumerate([28, 26, 12], start=1):
        summary.column_dimensions[get_column_letter(index)].width = width
    return _to_buffer(workbook)


def audit_xlsx(rows: list) -> io.BytesIO:
    """Журнал действий пользователей в формате Excel."""
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "Журнал действий"
    sheet.append(["Дата и время", "Пользователь", "Роль", "Действие", "Описание", "IP-адрес"])
    for row in rows:
        sheet.append([row.created_at.strftime("%d.%m.%Y %H:%M:%S"), row.user.login if row.user else "",
                      row.user.role.title if row.user else "", row.action, row.details, row.ip_address])
    _style_header(sheet, 6)
    for index, width in enumerate([20, 16, 16, 20, 80, 16], start=1):
        sheet.column_dimensions[get_column_letter(index)].width = width
    sheet.auto_filter.ref = sheet.dimensions
    return _to_buffer(workbook)
