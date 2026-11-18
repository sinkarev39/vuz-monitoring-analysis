# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль queries — запросы к базе данных для страниц веб-интерфейса.

Здесь сосредоточена выборка данных: показатели организации по годам, медианы,
последние прогнозы, сводные счётчики. Обработчики страниц получают готовые
структуры и занимаются только отображением.
"""

import json

import pandas as pd
from sqlalchemy import func, select

from vuzmon.config import MAIN_INDICATOR_TITLES
from vuzmon.db.models import (Indicator, IndicatorMedian, IndicatorValue, Prediction, Region,
                              University)
from vuzmon.ml.dataset import class_from_count, target_indicators


def name_filter(search: str):
    """
    Условие поиска организации по части названия без учёта регистра.
    Сравнение идёт с заранее подготовленным полем search_name, поэтому
    результат не зависит от того, как СУБД обращается с кириллицей.
    """
    return University.search_name.contains(search.strip().lower(), autoescape=True)


def indicators_by_column(db) -> dict:
    """Справочник показателей: имя столбца -> объект Indicator."""
    return {indicator.column: indicator for indicator in db.scalars(select(Indicator))}


def indicator_titles(db) -> dict:
    """Имя столбца -> название показателя (для подписей признаков модели)."""
    return {indicator.column: indicator.name for indicator in db.scalars(select(Indicator))}


def university_values(db, university_id: int) -> pd.DataFrame:
    """Все значения показателей организации: строки — годы, столбцы — показатели."""
    rows = db.execute(
        select(IndicatorValue.year, Indicator.column, IndicatorValue.value)
        .join(Indicator, Indicator.id == IndicatorValue.indicator_id)
        .where(IndicatorValue.university_id == university_id)).all()
    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows, columns=["year", "column", "value"])
    return frame.pivot(index="year", columns="column", values="value").sort_index()


def university_medians(db, university: University) -> dict:
    """
    Медианы основных показателей для организации.
    Возвращает словарь {(год, столбец): {'rf': ..., 'region': ..., 'founder': ...}}.
    """
    rows = db.execute(
        select(IndicatorMedian.year, Indicator.column, IndicatorMedian.level, IndicatorMedian.value)
        .join(Indicator, Indicator.id == IndicatorMedian.indicator_id)
        .where((IndicatorMedian.level == "rf")
               | ((IndicatorMedian.level == "region") & (IndicatorMedian.region_id == university.region_id))
               | ((IndicatorMedian.level == "founder")
                  & (IndicatorMedian.founder_group_id == university.founder_group_id)))).all()
    result = {}
    for year, column, level, value in rows:
        result.setdefault((year, column), {})[level] = value
    return result


def main_indicator_table(values: pd.DataFrame, medians: dict, is_branch: bool) -> dict:
    """
    Таблица основных показателей для карточки организации.

    Выполнение показателя оценивается по порогу самой организации (столбцы
    med_e1 ... med_e8): для E.8 медиана зависит от отраслевой специфики.
    Если порог организации неизвестен, берётся общая медиана по России.
    """
    codes = [code for code in target_indicators(is_branch) if code in values.columns]
    years = [int(year) for year in values.index if values.loc[year, codes].notna().any()] if codes else []
    rows, counts = [], {}
    for code in codes:
        cells = []
        for year in years:
            value = values.loc[year, code]
            value = None if pd.isna(value) else float(value)
            median = values.loc[year, f"med_{code}"] if f"med_{code}" in values.columns else None
            if median is None or pd.isna(median):
                median = medians.get((year, code), {}).get("rf")
            passed = None
            if median is not None and value is not None:
                passed = value >= median and value > 0
                counts[year] = counts.get(year, 0) + int(passed)
            elif median is not None:
                counts.setdefault(year, 0)
            cells.append({"year": year, "value": value, "median": median, "passed": passed})
        last = medians.get((years[-1], code), {}) if years else {}
        rows.append({"code": code, "title": MAIN_INDICATOR_TITLES[code], "cells": cells,
                     "median_region": last.get("region"), "median_founder": last.get("founder")})
    classes = {year: class_from_count(count) for year, count in counts.items()}
    return {"years": years, "rows": rows, "counts": counts, "classes": classes}


def latest_predictions(db, base_year: int = None) -> dict:
    """Последние прогнозы: код организации -> словарь с классом и вероятностями."""
    if base_year is None:
        base_year = db.scalar(select(func.max(Prediction.base_year)))
    if base_year is None:
        return {}
    result = {}
    for prediction in db.scalars(select(Prediction).where(Prediction.base_year == base_year)):
        result[prediction.university_id] = {
            "predicted_class": prediction.predicted_class, "target_year": prediction.target_year,
            "probabilities": json.loads(prediction.probabilities_json),
            "anomaly_score": prediction.anomaly_score, "cluster": prediction.cluster,
            "base_year": prediction.base_year, "created_at": prediction.created_at}
    return result


def current_classes(prepared, year: int = None) -> dict:
    """Фактический класс организаций в указанном (по умолчанию последнем) году мониторинга."""
    meta = prepared.samples.meta
    year = year or int(meta["year"].max())
    part = meta[meta["year"] == year]
    return {int(inst_id): (None if pd.isna(value) else int(value))
            for inst_id, value in zip(part["inst_id"], part["current_class"])}


def class_counts_by_year(prepared) -> tuple:
    """Число организаций каждого класса по годам мониторинга: (годы, матрица счётчиков)."""
    meta = prepared.samples.meta.dropna(subset=["current_class"])
    table = meta.groupby(["year", "current_class"]).size().unstack(fill_value=0)
    table = table.reindex(columns=[0, 1, 2], fill_value=0)
    return [int(year) for year in table.index], table.to_numpy()


def database_summary(db) -> dict:
    """Сводные счётчики по базе данных для страницы обзора."""
    years = db.execute(select(func.min(IndicatorValue.year), func.max(IndicatorValue.year))).one()
    return {
        "universities": db.scalar(select(func.count()).select_from(University)),
        "heads": db.scalar(select(func.count()).select_from(University).where(University.is_branch.is_(False))),
        "regions": db.scalar(select(func.count()).select_from(Region)),
        "values_count": db.scalar(select(func.count()).select_from(IndicatorValue)),
        "indicators": db.scalar(select(func.count()).select_from(Indicator)),
        "year_min": years[0], "year_max": years[1],
    }
