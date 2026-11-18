# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль preprocessing — подготовка табличных признаков для нейронной сети.

Шаги преобразования (параметры подбираются ТОЛЬКО по обучающей части):
  1) логарифмирование log(1 + x) признаков с «тяжёлым хвостом» (численности,
     площади, доходы различаются между вузами на порядки);
  2) заполнение пропусков медианой и добавление флага «значение отсутствовало»;
  3) стандартизация z = (x - среднее) / стандартное отклонение;
  4) ограничение выбросов диапазоном [-clip; clip].
Состояние преобразователя сохраняется вместе с моделью в виде обычных
списков, поэтому файл модели не зависит от версии сторонних библиотек.
"""

import numpy as np
import pandas as pd


class TabularPreprocessor:
    """Преобразователь таблицы признаков в числовую матрицу float32."""

    def __init__(self, skew_threshold: float = 2.0, missing_flag_share: float = 0.02, clip: float = 5.0):
        self.skew_threshold = skew_threshold          # порог асимметрии для логарифмирования
        self.missing_flag_share = missing_flag_share  # доля пропусков, начиная с которой добавляется флаг
        self.clip = clip                              # ограничение стандартизованных значений
        self.columns = []        # исходные признаки в фиксированном порядке
        self.log_columns = []    # признаки, к которым применяется log(1 + x)
        self.flag_columns = []   # признаки, для которых создаётся флаг пропуска
        self.medians = {}
        self.means = {}
        self.stds = {}

    # ------------------------------------------------------------ обучение
    def fit(self, frame: pd.DataFrame) -> "TabularPreprocessor":
        """Вычисляет параметры преобразования по обучающей выборке."""
        self.columns = list(frame.columns)
        self.log_columns, self.flag_columns = [], []
        self.medians, self.means, self.stds = {}, {}, {}
        for column in self.columns:
            values = frame[column].astype("float64")
            observed = values.dropna()
            # Логарифмируем неотрицательные признаки с сильной правой асимметрией
            if len(observed) > 10 and observed.min() >= 0 and observed.nunique() > 2 \
                    and observed.skew() > self.skew_threshold:
                self.log_columns.append(column)
                values = np.log1p(values)
                observed = values.dropna()
            if values.isna().mean() >= self.missing_flag_share:
                self.flag_columns.append(column)
            median = float(observed.median()) if len(observed) else 0.0
            filled = values.fillna(median)
            std = float(filled.std(ddof=0))
            self.medians[column] = median
            self.means[column] = float(filled.mean())
            self.stds[column] = std if std > 1e-9 else 1.0   # константный признак не делим на ноль
        return self

    # ------------------------------------------------------ преобразование
    @property
    def feature_names(self) -> list:
        """Имена столбцов итоговой матрицы (признаки + флаги пропусков)."""
        return self.columns + [f"{column}_isna" for column in self.flag_columns]

    def transform(self, frame: pd.DataFrame) -> np.ndarray:
        """Применяет подобранное преобразование к новой таблице признаков."""
        frame = frame.reindex(columns=self.columns)   # отсутствующие столбцы станут пропусками
        result = np.empty((len(frame), len(self.feature_names)), dtype=np.float32)
        for index, column in enumerate(self.columns):
            values = frame[column].to_numpy(dtype="float64", na_value=np.nan)
            if column in self.log_columns:
                values = np.log1p(np.clip(values, 0.0, None))
            values = np.where(np.isfinite(values), values, self.medians[column])
            values = (values - self.means[column]) / self.stds[column]
            result[:, index] = np.clip(values, -self.clip, self.clip)
        for offset, column in enumerate(self.flag_columns):
            values = frame[column].to_numpy(dtype="float64", na_value=np.nan)
            result[:, len(self.columns) + offset] = (~np.isfinite(values)).astype(np.float32)
        return result

    def fit_transform(self, frame: pd.DataFrame) -> np.ndarray:
        return self.fit(frame).transform(frame)

    # ------------------------------------------------- сохранение состояния
    def state_dict(self) -> dict:
        """Состояние в виде простых типов Python (пригодно для torch.save и JSON)."""
        return {
            "skew_threshold": self.skew_threshold, "missing_flag_share": self.missing_flag_share,
            "clip": self.clip, "columns": list(self.columns), "log_columns": list(self.log_columns),
            "flag_columns": list(self.flag_columns),
            "medians": [self.medians[c] for c in self.columns],
            "means": [self.means[c] for c in self.columns],
            "stds": [self.stds[c] for c in self.columns],
        }

    @classmethod
    def from_state(cls, state: dict) -> "TabularPreprocessor":
        """Восстанавливает преобразователь из сохранённого состояния."""
        instance = cls(state["skew_threshold"], state["missing_flag_share"], state["clip"])
        instance.columns = list(state["columns"])
        instance.log_columns = list(state["log_columns"])
        instance.flag_columns = list(state["flag_columns"])
        instance.medians = dict(zip(instance.columns, state["medians"]))
        instance.means = dict(zip(instance.columns, state["means"]))
        instance.stds = dict(zip(instance.columns, state["stds"]))
        return instance
