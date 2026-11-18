# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль loader — загрузка набора данных мониторинга (CSV) в базу данных.

Разнородные таблицы сборщика (паспорта организаций, основные показатели,
дополнительные характеристики, медианы) приводятся к нормализованной
структуре: справочники + таблица фактов «организация — год — показатель».
"""

import re

import pandas as pd
from sqlalchemy import delete, insert, select

from vuzmon.db.models import (FederalDistrict, FounderGroup, Indicator, IndicatorMedian,
                              IndicatorValue, Region, University)

# Профильные характеристики организации, которые сборщик кладёт в паспорт
PROFILE_INDICATORS = {
    "ugn_count": ("Число реализуемых укрупнённых групп направлений подготовки (специальностей)", "ед."),
    "ugn_top_share": ("Доля крупнейшей УГН(С) в приведённом контингенте студентов", "%"),
    "sf_math": ("Приведённый контингент: математические и естественные науки", "чел."),
    "sf_eng": ("Приведённый контингент: инженерное дело, технологии и технические науки", "чел."),
    "sf_med": ("Приведённый контингент: здравоохранение и медицинские науки", "чел."),
    "sf_agro": ("Приведённый контингент: сельское хозяйство и сельскохозяйственные науки", "чел."),
    "sf_soc": ("Приведённый контингент: науки об обществе", "чел."),
    "sf_edu": ("Приведённый контингент: образование и педагогические науки", "чел."),
    "sf_hum": ("Приведённый контингент: гуманитарные науки", "чел."),
    "sf_art": ("Приведённый контингент: искусство и культура", "чел."),
    "sf_def": ("Приведённый контингент: оборона и безопасность государства, военные науки", "чел."),
}

_QUOTED_RE = re.compile(r"[«\"“](.+)[»\"”]")
_LEGAL_FORM_RE = re.compile(r"^.*?высшего\s+(?:профессионального\s+)?образования\s*", re.I | re.S)
CHUNK_SIZE = 20000   # размер пакета при массовой вставке строк


def make_short_name(name: str, is_branch: bool = False, city: str = "") -> str:
    """
    Краткое название организации для таблиц и графиков.

    Из полного названия убирается организационно-правовая форма; если
    название взято в кавычки, используется текст в кавычках. Для филиала
    добавляется пометка с городом.
    """
    name = (name or "").strip()
    match = _QUOTED_RE.search(name)
    short = match.group(1).strip() if match else _LEGAL_FORM_RE.sub("", name).strip() or name
    short = short[:1].upper() + short[1:] if short else name
    if is_branch:
        short = f"{short} — филиал ({city})" if city else f"{short} — филиал"
    return short[:255]


def consolidate_main(main: pd.DataFrame) -> pd.DataFrame:
    """
    Оставляет по одной строке основных показателей на организацию и год.

    Одно и то же значение публикуется на страницах нескольких лет (в составе
    пятилетней динамики). Приоритет отдаётся странице «своего» года, а при её
    отсутствии — ближайшей более поздней.
    """
    frame = main.copy()
    frame["_other_year"] = (frame["page_year"] != frame["value_year"]).astype(int)
    frame = frame.sort_values(["inst_id", "value_year", "_other_year", "page_year"])
    frame = frame.drop_duplicates(["inst_id", "value_year"], keep="first")
    return frame.drop(columns=["_other_year", "page_year"]).rename(columns={"value_year": "year"}) \
        .reset_index(drop=True)


def _most_frequent(series: pd.Series):
    """Наиболее часто встречающееся значение (медианы одинаковы внутри группы)."""
    values = series.dropna()
    return values.mode().iloc[0] if not values.empty else None


def _insert_chunks(session, table, rows: list) -> None:
    """Массовая вставка строк пакетами — в десятки раз быстрее построчной."""
    for start in range(0, len(rows), CHUNK_SIZE):
        session.execute(insert(table), rows[start:start + CHUNK_SIZE])


def load_monitoring_frames(session, frames: dict, progress=None) -> dict:
    """
    Загружает набор данных мониторинга в базу. Прежние значения показателей
    и медианы заменяются, справочники обновляются.

    Возвращает словарь с количеством загруженных строк по таблицам.
    """
    notify = progress or (lambda message: None)
    institutions, extra = frames["institutions"], frames["extra"]
    main = consolidate_main(frames["main"])
    stats = {}

    # ---------------------------------------------------------- справочники
    notify("Обновление справочников округов, регионов и ведомств")
    districts = {d.name: d for d in session.scalars(select(FederalDistrict))}
    for name in sorted(institutions["federal_district"].dropna().unique()):
        if name not in districts:
            districts[name] = FederalDistrict(name=name)
            session.add(districts[name])
    session.flush()

    latest = institutions.sort_values("year").drop_duplicates("inst_id", keep="last")
    region_rows = latest.drop_duplicates("region_id")
    for row in region_rows.itertuples():
        session.merge(Region(id=int(row.region_id), name=row.region,
                             district_id=districts[row.federal_district].id))

    founders = {f.name: f for f in session.scalars(select(FounderGroup))}
    for name in sorted(n for n in institutions["founder_group"].dropna().unique() if n):
        if name not in founders:
            founders[name] = FounderGroup(name=name)
            session.add(founders[name])
    session.flush()

    # ---------------------------------------------------------- организации
    notify("Обновление перечня организаций")
    for row in latest.itertuples():
        is_branch = bool(row.is_branch)
        city = row.city if isinstance(row.city, str) else ""
        founder = founders.get(row.founder_group) if isinstance(row.founder_group, str) else None
        session.merge(University(
            id=int(row.inst_id), name=row.name, short_name=make_short_name(row.name, is_branch, city),
            is_branch=is_branch, head_id=int(row.head_id) if pd.notna(row.head_id) else None,
            region_id=int(row.region_id), founder_group_id=founder.id if founder else None,
            city=city, address=row.address if isinstance(row.address, str) else "",
            website=row.website if isinstance(row.website, str) else "",
            specifics=row.specifics if isinstance(row.specifics, str) else "",
            longitude=row.longitude if pd.notna(row.longitude) else None,
            latitude=row.latitude if pd.notna(row.latitude) else None, last_year=int(row.year)))
    stats["universities"] = len(latest)

    # ------------------------------------------------- справочник показателей
    indicators = {i.column: i for i in session.scalars(select(Indicator))}
    for row in frames["dictionary"].itertuples():
        if row.column not in indicators:
            kind = "main" if row.column.startswith("e") else "extra"
            indicators[row.column] = Indicator(column=row.column, code=str(row.code), kind=kind,
                                               section=row.section, name=row.name,
                                               unit=row.unit if isinstance(row.unit, str) else "")
            session.add(indicators[row.column])
    for column, (name, unit) in PROFILE_INDICATORS.items():
        if column not in indicators:
            indicators[column] = Indicator(column=column, code=column, kind="profile",
                                           section="Профиль подготовки", name=name, unit=unit)
            session.add(indicators[column])
    session.flush()
    stats["indicators"] = len(indicators)

    # ------------------------------------------------- значения показателей
    notify("Загрузка значений показателей")
    session.execute(delete(IndicatorValue))

    def melt(frame: pd.DataFrame, year_column: str) -> list:
        """Переводит «широкую» таблицу в строки (организация, год, показатель, значение)."""
        columns = [c for c in frame.columns if c in indicators]
        long = frame.melt(id_vars=["inst_id", year_column], value_vars=columns,
                          var_name="column", value_name="value").dropna(subset=["value"])
        long["indicator_id"] = long["column"].map(lambda c: indicators[c].id)
        return [{"university_id": int(u), "year": int(y), "indicator_id": int(i), "value": float(v)}
                for u, y, i, v in zip(long["inst_id"], long[year_column], long["indicator_id"], long["value"])]

    # Основные показатели известны и для организаций, которые позже выбыли из
    # мониторинга; в таблицу фактов попадают только организации из справочника
    known_ids = set(latest["inst_id"])
    value_rows = melt(main[main["inst_id"].isin(known_ids)], "year")
    value_rows += melt(extra, "year")
    value_rows += melt(institutions, "year")
    _insert_chunks(session, IndicatorValue, value_rows)
    stats["indicator_values"] = len(value_rows)

    # ------------------------------------------------------------- медианы
    notify("Загрузка медианных значений")
    session.execute(delete(IndicatorMedian))
    medians = frames["medians"].merge(institutions[["year", "inst_id", "region_id", "founder_group"]],
                                      on=["year", "inst_id"], how="left")
    median_rows = []
    for (year, code), group in medians.groupby(["year", "code"]):
        median_rows.append({"year": int(year), "indicator_id": indicators[code].id, "level": "rf",
                            "region_id": None, "founder_group_id": None,
                            "value": _most_frequent(group["median_rf"])})
    for (year, code, region_id), group in medians.groupby(["year", "code", "region_id"]):
        median_rows.append({"year": int(year), "indicator_id": indicators[code].id, "level": "region",
                            "region_id": int(region_id), "founder_group_id": None,
                            "value": _most_frequent(group["median_region"])})
    for (year, code, founder_name), group in medians.groupby(["year", "code", "founder_group"]):
        if founder_name in founders:
            median_rows.append({"year": int(year), "indicator_id": indicators[code].id, "level": "founder",
                                "region_id": None, "founder_group_id": founders[founder_name].id,
                                "value": _most_frequent(group["median_founder"])})
    _insert_chunks(session, IndicatorMedian, median_rows)
    stats["indicator_medians"] = len(median_rows)

    session.commit()
    notify("Загрузка завершена")
    return stats
