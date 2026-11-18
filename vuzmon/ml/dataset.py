# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль dataset — формирование обучающей выборки из данных мониторинга.

Постановка задачи. По сведениям об организации за год t (основные показатели,
их динамика и 56 дополнительных характеристик) требуется спрогнозировать класс
результативности в следующем мониторинге (год t+1):

    0 — «зона риска»      (не ниже медианы по России 0–1 показателя из 5);
    1 — «средний уровень» (2–3 показателя);
    2 — «лидеры»          (4–5 показателей).

Объект выборки — пара «организация, год». Разбиение на обучающую,
валидационную и тестовую части выполняется по времени, чтобы модель
проверялась на более поздних данных, чем те, на которых обучалась.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from vuzmon.tables import consolidate_main, institution_medians

# Показатели, по которым определяется класс результативности
HEAD_TARGET_INDICATORS = ("e1", "e2", "e3", "e4", "e8")     # головные организации
BRANCH_TARGET_INDICATORS = ("e1", "e2", "e4", "e7", "e8")   # филиалы (вместо E.3 — контингент E.7)

CLASS_NAMES = ("Зона риска", "Средний уровень", "Лидеры")
CLASS_SHORT = ("риск", "средний", "лидер")

# Отрасли наук: столбцы паспорта организации
SCIENCE_FIELDS = ("sf_math", "sf_eng", "sf_med", "sf_agro", "sf_soc", "sf_edu", "sf_hum", "sf_art", "sf_def")

# Укрупнение ведомственной принадлежности: ключевое слово -> группа
FOUNDER_CATEGORIES = (
    ("науки и высшего образования", "minobr"),
    ("здравоохранения", "minzdrav"),
    ("сельского хозяйства", "minselhoz"),
    ("культуры", "mincult"),
    ("просвещения", "minpros"),
    ("частн", "private"),
)
FOUNDER_CATEGORY_CODES = ("minobr", "minzdrav", "minselhoz", "mincult", "minpros", "private", "regional",
                          "other_federal")
# Федеральные округа в том написании, которое используется на сайте мониторинга
FEDERAL_DISTRICTS = ("ЦФО", "СЗФО", "ЮФО", "СКФО", "ПФО", "УФО", "СФО", "ДФО")
# Отраслевая специфика деятельности: ключевое слово -> группа
SPECIFICS_CATEGORIES = (
    ("медицин", "spec_med"), ("сельскохоз", "spec_agro"), ("творческ", "spec_art"),
    ("спортив", "spec_sport"), ("транспорт", "spec_transport"),
)


def class_from_count(count):
    """Класс результативности по числу показателей не ниже медианы (0..5)."""
    if pd.isna(count):
        return np.nan
    return 0 if count <= 1 else (1 if count <= 3 else 2)


def target_indicators(is_branch: bool) -> tuple:
    """Набор показателей, по которым оценивается организация данного типа."""
    return BRANCH_TARGET_INDICATORS if is_branch else HEAD_TARGET_INDICATORS


def rf_medians(medians: pd.DataFrame) -> pd.DataFrame:
    """
    Сводная таблица медиан по России: строки — годы, столбцы — показатели.
    Используется для справки; при расчётах берутся медианы конкретной организации.
    """
    table = medians.groupby(["year", "code"])["median_rf"] \
        .agg(lambda s: s.dropna().mode().iloc[0] if s.notna().any() else np.nan)
    return table.unstack("code")


def founder_category(name) -> str:
    """Укрупнённая группа учредителя: Минобрнауки, Минздрав, частные и т. д."""
    text = name.lower() if isinstance(name, str) else ""
    for keyword, category in FOUNDER_CATEGORIES:
        if keyword in text:
            return category
    if "субъект" in text or "муниципал" in text:
        return "regional"
    return "other_federal"


def _safe_ratio(values: pd.Series, base) -> pd.Series:
    """Отношение показателя к медиане; при нулевой медиане результат не определён."""
    base = pd.Series(base, index=values.index, dtype="float64").where(lambda b: b > 0)
    return values / base


def _growth(current: pd.Series, previous: pd.Series) -> pd.Series:
    """Относительный прирост за год, ограниченный диапазоном [-1; 3]."""
    previous = previous.where(previous.abs() > 1e-9)
    return ((current - previous) / previous.abs()).clip(-1.0, 3.0)


@dataclass
class SampleSet:
    """Готовая выборка: признаки, целевой класс и сопроводительные сведения."""
    features: pd.DataFrame     # матрица «объект — признак»
    target: pd.Series          # класс следующего года (NaN, если ещё не известен)
    meta: pd.DataFrame         # inst_id, year, name, is_branch, current_class ...

    def subset(self, mask) -> "SampleSet":
        """Часть выборки по логической маске."""
        mask = np.asarray(mask)
        return SampleSet(self.features[mask].reset_index(drop=True),
                         self.target[mask].reset_index(drop=True),
                         self.meta[mask].reset_index(drop=True))

    def labeled(self) -> "SampleSet":
        """Только объекты с известным классом следующего года."""
        return self.subset(self.target.notna().to_numpy())

    def __len__(self) -> int:
        return len(self.features)


def build_panel(frames: dict) -> pd.DataFrame:
    """
    Сводит все таблицы набора данных в панель «организация — год».

    К паспорту организации присоединяются основные показатели текущего и двух
    предыдущих лет, дополнительные характеристики и медианы по России.
    """
    inst = frames["institutions"].copy()
    main = consolidate_main(frames["main"])
    medians = institution_medians(frames["medians"])
    e_columns = [c for c in main.columns if c.startswith("e")]
    # Не у каждой организации есть все показатели — недостающие столбцы медиан создаём пустыми
    for column in e_columns:
        if f"med_{column}" not in medians.columns:
            medians[f"med_{column}"] = np.nan

    panel = inst.merge(main, on=["inst_id", "year"], how="left")
    # Значения предыдущих лет: сдвигаем год и присоединяем с суффиксом
    for lag in (1, 2):
        shifted = main.copy()
        shifted["year"] = shifted["year"] + lag
        panel = panel.merge(shifted.rename(columns={c: f"{c}_lag{lag}" for c in e_columns}),
                            on=["inst_id", "year"], how="left")
    # Значения следующего года нужны только для вычисления целевого класса
    following = main.copy()
    following["year"] = following["year"] - 1
    panel = panel.merge(following.rename(columns={c: f"{c}_next" for c in e_columns}),
                        on=["inst_id", "year"], how="left")
    # Медианы текущего и следующего года (со страницы самой организации)
    panel = panel.merge(medians, on=["inst_id", "year"], how="left")
    next_medians = medians.rename(columns={f"med_{c}": f"mednext_{c}" for c in e_columns})
    next_medians["year"] = next_medians["year"] - 1
    panel = panel.merge(next_medians, on=["inst_id", "year"], how="left")
    panel = panel.merge(frames["extra"], on=["inst_id", "year"], how="left")

    # Отраслевая специфика публикуется не во все годы: берём известное значение организации
    panel["specifics"] = panel["specifics"].where(panel["specifics"].astype(str).str.len() > 3)
    panel = panel.sort_values(["inst_id", "year"])
    panel["specifics"] = panel.groupby("inst_id")["specifics"].transform(lambda s: s.ffill().bfill())
    return panel.reset_index(drop=True)


def _pass_count(panel: pd.DataFrame, value_suffix: str, median_prefix: str) -> pd.Series:
    """Число показателей организации, которые не ниже медианы по России."""
    counts = pd.Series(0.0, index=panel.index)
    known = pd.Series(False, index=panel.index)
    for is_branch, indicators in ((0, HEAD_TARGET_INDICATORS), (1, BRANCH_TARGET_INDICATORS)):
        rows = panel["is_branch"] == is_branch
        for code in indicators:
            value = panel.loc[rows, f"{code}{value_suffix}"]
            median = panel.loc[rows, f"{median_prefix}{code}"]
            # Пропуск значения трактуется как невыполнение показателя
            counts.loc[rows] += ((value >= median) & (value > 0)).astype(float)
            known.loc[rows] |= value.notna() & median.notna()
    return counts.where(known)


def build_samples(frames: dict, include_branches: bool = False) -> SampleSet:
    """
    Строит выборку для обучения и прогнозирования.

    include_branches — включать ли филиалы (по умолчанию модель строится
    для головных организаций, у которых единый набор показателей).
    """
    panel = build_panel(frames)
    if not include_branches:
        panel = panel[panel["is_branch"] == 0].reset_index(drop=True)

    features = {}   # имя признака -> столбец значений; таблица создаётся в конце одним действием
    all_codes = sorted(set(HEAD_TARGET_INDICATORS) | (set(BRANCH_TARGET_INDICATORS) if include_branches else set()))

    # --- 1. Основные показатели: положение относительно медианы и динамика
    for code in all_codes:
        median = panel[f"med_{code}"]
        features[f"{code}_ratio"] = _safe_ratio(panel[code], median)
        features[f"{code}_ratio_lag1"] = _safe_ratio(panel[f"{code}_lag1"], median)
        features[f"{code}_ratio_lag2"] = _safe_ratio(panel[f"{code}_lag2"], median)
        features[f"{code}_growth"] = _growth(panel[code], panel[f"{code}_lag1"])
        features[f"{code}_pass"] = ((panel[code] >= median) & (panel[code] > 0)).astype(float)
    current_count = _pass_count(panel, "", "med_")
    features["pass_count"] = current_count

    # --- 2. Дополнительные характеристики мониторинга (столбцы p1 ... p56)
    for column in [c for c in frames["extra"].columns if c.startswith("p")]:
        features[column] = panel[column]

    # --- 3. Профиль подготовки: доли контингента по отраслям наук
    contingent = panel[list(SCIENCE_FIELDS)].sum(axis=1)
    features["contingent_total"] = contingent
    for column in SCIENCE_FIELDS:
        features[f"{column}_share"] = (panel[column] / contingent.where(contingent > 0)).fillna(0.0)
    features["ugn_count"] = panel["ugn_count"]
    features["ugn_top_share"] = panel["ugn_top_share"]

    # --- 4. Производные относительные показатели (на одного студента / работника)
    students = panel["p1"].where(panel["p1"] > 0)
    staff = panel["p22"].where(panel["p22"] > 0)
    features["income_per_student"] = panel["p48"] / students
    features["students_per_teacher"] = students / staff
    features["pc_per_100_students"] = panel["p45"] / students * 100.0
    features["area_per_student"] = panel["p39"] / students
    features["postgrad_per_100_students"] = panel["p17"] / students * 100.0

    # --- 5. Категориальные признаки в виде индикаторных столбцов
    for district in FEDERAL_DISTRICTS:
        features[f"fo_{district}"] = (panel["federal_district"] == district).astype(float)
    categories = panel["founder_group"].map(founder_category)
    for category in FOUNDER_CATEGORY_CODES:
        features[f"founder_{category}"] = (categories == category).astype(float)
    specifics = panel["specifics"].fillna("").astype(str).str.lower()
    for keyword, column in SPECIFICS_CATEGORIES:
        features[column] = specifics.str.contains(keyword).astype(float)
    if include_branches:
        features["is_branch"] = panel["is_branch"].astype(float)
    features = pd.DataFrame(features, index=panel.index)

    # --- Целевой класс: положение организации в мониторинге следующего года
    next_count = _pass_count(panel, "_next", "mednext_")
    target = next_count.map(class_from_count)

    meta = panel[["inst_id", "year", "name", "is_branch", "region", "federal_district",
                  "founder_group"]].copy()
    meta["current_count"] = current_count
    meta["current_class"] = current_count.map(class_from_count)
    meta["next_count"] = next_count
    return SampleSet(features.reset_index(drop=True).astype("float64"),
                     target.reset_index(drop=True), meta.reset_index(drop=True))


def temporal_split(samples: SampleSet, val_year: int = None, test_year: int = None) -> dict:
    """
    Разбиение по времени. По умолчанию последний год с известным ответом —
    тестовый, предпоследний — валидационный, все более ранние — обучающие.
    Объекты последнего года мониторинга (ответ ещё не известен) образуют
    часть 'future' — для них система строит прогноз.
    """
    labeled_years = sorted(samples.meta.loc[samples.target.notna(), "year"].unique())
    if len(labeled_years) < 3:
        raise ValueError("Для разбиения по времени нужны данные минимум за четыре года мониторинга")
    test_year = test_year or labeled_years[-1]
    val_year = val_year or labeled_years[-2]
    year = samples.meta["year"]
    known = samples.target.notna()
    return {
        "train": samples.subset(known & (year < val_year)),
        "val": samples.subset(known & (year == val_year)),
        "test": samples.subset(known & (year == test_year)),
        "future": samples.subset(~known & (year == year.max())),
    }
