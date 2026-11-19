# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль validation — проверка набора данных, полученного извне (из файлов
пользователя или по ссылке), до того как он будет сохранён в системе.

Набор, собранный сборщиком с сайта мониторинга, формируется программой и всегда
имеет правильные типы. Файлы, загруженные пользователем, могли быть изменены
вручную: текст в числовом столбце или пропущенный год приводят к сбою уже при
построении выборки, когда причина не очевидна. Поэтому содержимое проверяется
сразу, а замечание называет файл, столбец, строку и само значение.
"""

import pandas as pd

from vuzmon import config
from vuzmon.collect.crawler import DATASET_FILES

# Столбцы, которые обязаны быть в таблицах набора
REQUIRED_COLUMNS = {
    "institutions": {"year", "inst_id", "name", "is_branch", "region_id", "region", "federal_district",
                     "founder_group"},
    "main": {"inst_id", "page_year", "value_year", "e1", "e2", "e4", "e8"},
    "medians": {"year", "inst_id", "code", "median_rf"},
    "extra": {"year", "inst_id", "p1", "p22", "p48"},
    "dictionary": {"column", "code", "section", "name"},
}

# Ключевые столбцы: целые числа без пропусков
INTEGER_COLUMNS = {
    "institutions": ("year", "inst_id", "region_id", "is_branch"),
    "main": ("inst_id", "page_year", "value_year"),
    "medians": ("year", "inst_id"),
    "extra": ("year", "inst_id"),
}
# Столбцы с годом: значение должно быть правдоподобным годом мониторинга
YEAR_COLUMNS = {"institutions": ("year",), "main": ("page_year", "value_year"), "medians": ("year",),
                "extra": ("year",)}
YEAR_MIN, YEAR_MAX = 2000, 2100
MAX_PROBLEMS = 8      # больше замечаний не собираем: пользователю достаточно первых


def _numeric_columns(table: str, frame: pd.DataFrame) -> list:
    """Столбцы таблицы, в которых допускаются только числа (пропуски разрешены)."""
    if table == "main":
        return [column for column in config.MAIN_INDICATORS.values() if column in frame.columns]
    if table == "medians":
        return [column for column in ("median_rf", "median_region", "median_founder") if column in frame.columns]
    if table == "extra":
        return [column for column in frame.columns if column not in ("year", "inst_id")]
    return []


def _first_bad(series: pd.Series, bad: pd.Series) -> str:
    """Описание первого некорректного значения: номер строки файла и само значение."""
    position = int(bad.to_numpy().argmax())
    value = series.iloc[position]
    shown = "пусто" if pd.isna(value) else f"«{str(value)[:40]}»"
    return f"в строке {position + 2} указано {shown}"       # +2: нумерация с единицы и строка заголовков


def check_frames(frames: dict) -> list:
    """
    Проверяет таблицы набора данных. Возвращает список замечаний на русском
    языке; пустой список означает, что набор можно сохранять.
    """
    problems = []
    for table, filename in DATASET_FILES.items():
        frame = frames[table]
        missing = REQUIRED_COLUMNS[table] - set(frame.columns)
        if missing:
            problems.append(f"В файле {filename} нет столбцов: {', '.join(sorted(missing))}")
            continue
        if frame.empty:
            problems.append(f"В файле {filename} нет ни одной записи")
            continue

        for column in INTEGER_COLUMNS.get(table, ()):
            numbers = pd.to_numeric(frame[column], errors="coerce")
            bad = numbers.isna() | (numbers % 1 != 0)
            if bad.any():
                problems.append(f"Файл {filename}, столбец «{column}»: ожидается целое число, "
                                f"{_first_bad(frame[column], bad)}")
            elif column in YEAR_COLUMNS.get(table, ()) and not numbers.between(YEAR_MIN, YEAR_MAX).all():
                problems.append(f"Файл {filename}, столбец «{column}»: год должен быть от {YEAR_MIN} до {YEAR_MAX}, "
                                f"{_first_bad(frame[column], ~numbers.between(YEAR_MIN, YEAR_MAX))}")
            elif column == "is_branch" and not numbers.isin([0, 1]).all():
                problems.append(f"Файл {filename}, столбец «is_branch»: допустимы значения 0 (головная "
                                f"организация) и 1 (филиал), {_first_bad(frame[column], ~numbers.isin([0, 1]))}")

        for column in _numeric_columns(table, frame):
            series = frame[column]
            if pd.api.types.is_numeric_dtype(series):
                continue
            bad = series.notna() & pd.to_numeric(series, errors="coerce").isna()
            if bad.any():
                problems.append(f"Файл {filename}, столбец «{column}»: ожидается число, {_first_bad(series, bad)}")

        if table == "institutions":
            empty = frame["name"].isna() | (frame["name"].astype(str).str.strip() == "")
            if empty.any():
                problems.append(f"Файл {filename}, столбец «name»: название организации не заполнено, "
                                f"{_first_bad(frame['name'], empty)}")
        if len(problems) >= MAX_PROBLEMS:
            break
    return problems[:MAX_PROBLEMS]
