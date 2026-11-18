# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль explain — объяснение решений модели.

  permutation_importance — важность признаков методом перестановок: значения
      одного признака перемешиваются между объектами, и измеряется, насколько
      упало качество модели. Метод не зависит от вида модели.
  local_contributions — вклад признаков в прогноз для одной организации:
      признак заменяется «типичным» значением (нулём после стандартизации),
      и фиксируется изменение вероятности предсказанного класса.
  feature_title — понятное человеку название признака.
"""

import re

import numpy as np

from vuzmon.config import MAIN_INDICATOR_TITLES
from vuzmon.ml import metrics

_SUFFIX_TITLES = {
    "ratio": "отношение к медиане по РФ",
    "ratio_lag1": "отношение к медиане, год назад",
    "ratio_lag2": "отношение к медиане, два года назад",
    "growth": "прирост за год",
    "pass": "выполнение (не ниже медианы)",
    "margin": "запас до медианы",
    "margin_lag1": "запас до медианы, год назад",
}
_ORDER_WORDS = ("наименьший", "второй по величине", "третий по величине", "четвёртый по величине", "наибольший")
_OTHER_TITLES = {
    "pass_count": "Число выполненных основных показателей в текущем году",
    "contingent_total": "Приведённый контингент студентов, всего",
    "ugn_count": "Число реализуемых УГН(С)",
    "ugn_top_share": "Доля крупнейшей УГН(С) в контингенте",
    "income_per_student": "Доходы вуза в расчёте на одного студента",
    "students_per_teacher": "Число студентов на одного преподавателя",
    "pc_per_100_students": "Персональных компьютеров на 100 студентов",
    "area_per_student": "Общая площадь зданий на одного студента",
    "postgrad_per_100_students": "Аспирантов и ординаторов на 100 студентов",
    "founder_minobr": "Учредитель: Минобрнауки России", "founder_minzdrav": "Учредитель: Минздрав России",
    "founder_minselhoz": "Учредитель: Минсельхоз России", "founder_mincult": "Учредитель: Минкультуры России",
    "founder_minpros": "Учредитель: Минпросвещения России", "founder_private": "Частная организация",
    "founder_regional": "Учредитель: субъект РФ", "founder_other_federal": "Учредитель: иной федеральный орган",
    "spec_med": "Специфика: медицинская", "spec_agro": "Специфика: сельскохозяйственная",
    "spec_art": "Специфика: творческая", "spec_sport": "Специфика: спортивная",
    "spec_transport": "Специфика: транспортная", "is_branch": "Филиал",
    "sf_math_share": "Доля контингента: математические и естественные науки",
    "sf_eng_share": "Доля контингента: инженерное дело и технологии",
    "sf_med_share": "Доля контингента: здравоохранение", "sf_agro_share": "Доля контингента: сельское хозяйство",
    "sf_soc_share": "Доля контингента: науки об обществе", "sf_edu_share": "Доля контингента: образование",
    "sf_hum_share": "Доля контингента: гуманитарные науки", "sf_art_share": "Доля контингента: искусство и культура",
    "sf_def_share": "Доля контингента: оборона и безопасность",
}


def feature_title(name: str, dictionary: dict = None) -> str:
    """
    Русское название признака.

    dictionary — словарь {имя столбца: название показателя} из справочника
    показателей мониторинга (для столбцов p1 ... p56).
    """
    dictionary = dictionary or {}
    if name.endswith("_isna"):
        return f"Нет данных: {feature_title(name[:-5], dictionary)}"
    if name in _OTHER_TITLES:
        return _OTHER_TITLES[name]
    match = re.match(r"^margin_sorted_(\d)$", name)
    if match and 1 <= int(match.group(1)) <= 5:
        return f"Запас до медианы: {_ORDER_WORDS[int(match.group(1)) - 1]} из пяти показателей"
    match = re.match(r"^(e\d)_(.+)$", name)
    if match and match.group(2) in _SUFFIX_TITLES:
        base = MAIN_INDICATOR_TITLES.get(match.group(1), match.group(1)).split(" (")[0]
        return f"{base}: {_SUFFIX_TITLES[match.group(2)]}"
    if name.startswith("fo_"):
        return f"Федеральный округ: {name[3:]}"
    if name in dictionary:
        return str(dictionary[name])
    return name


def permutation_importance(predict, features: np.ndarray, labels, feature_names: list,
                           n_repeats: int = 5, seed: int = 42) -> list:
    """
    Важность признаков методом перестановок.

    predict — функция, возвращающая матрицу вероятностей классов;
    результат — список словарей, отсортированный по убыванию важности:
    {'feature', 'importance' (падение macro-F1), 'std'}.
    """
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels, dtype=int)
    baseline = metrics.classification_report(labels, predict(features))["macro_f1"]
    rows = []
    for index, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            shuffled = features.copy()
            shuffled[:, index] = rng.permutation(shuffled[:, index])
            drops.append(baseline - metrics.classification_report(labels, predict(shuffled))["macro_f1"])
        rows.append({"feature": name, "importance": float(np.mean(drops)), "std": float(np.std(drops))})
    return sorted(rows, key=lambda row: row["importance"], reverse=True)


def local_contributions(predict, row: np.ndarray, feature_names: list, top: int = 8) -> list:
    """
    Вклад признаков в прогноз для одного объекта.

    Каждый признак поочерёдно заменяется нулём (средним значением после
    стандартизации); положительный вклад означает, что фактическое значение
    признака увеличивает вероятность предсказанного класса.
    """
    row = np.asarray(row, dtype=np.float32).reshape(1, -1)
    base = predict(row)[0]
    predicted = int(base.argmax())
    # Все варианты «с обнулённым признаком» считаем одним пакетом
    variants = np.repeat(row, row.shape[1], axis=0)
    variants[np.arange(row.shape[1]), np.arange(row.shape[1])] = 0.0
    changed = predict(variants)[:, predicted]
    contributions = base[predicted] - changed
    order = np.argsort(-np.abs(contributions))[:top]
    return [{"feature": feature_names[i], "contribution": float(contributions[i]), "value": float(row[0, i])}
            for i in order]
