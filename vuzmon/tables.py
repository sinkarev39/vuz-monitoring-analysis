# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль tables — общие преобразования таблиц набора данных, которые нужны
и загрузчику базы данных, и подсистеме машинного обучения.
"""

import pandas as pd


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


def institution_medians(medians: pd.DataFrame) -> pd.DataFrame:
    """
    Медианы по России в разрезе организаций: строка — (организация, год),
    столбцы — med_e1 ... med_e8.

    Медиана берётся со страницы самой организации, потому что для показателя
    E.8 она зависит от отраслевой специфики (у медицинских, творческих,
    спортивных вузов дополнительный показатель свой).
    """
    table = medians.pivot_table(index=["inst_id", "year"], columns="code", values="median_rf",
                                aggfunc="first")
    return table.add_prefix("med_").reset_index()
