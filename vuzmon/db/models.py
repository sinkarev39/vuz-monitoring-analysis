# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль models — описание таблиц базы данных средствами SQLAlchemy ORM.

База данных состоит из трёх групп таблиц:
  * справочники и данные мониторинга (округа, регионы, ведомства, организации,
    показатели, значения показателей, медианы);
  * результаты интеллектуального анализа (наборы данных, эксперименты,
    метрики по эпохам, прогнозы, заметки аналитиков);
  * служебные таблицы (роли, пользователи, журнал действий, настройки).
ORM позволяет без изменения кода перейти с SQLite на PostgreSQL:
достаточно указать другой адрес подключения в настройках.
"""

from datetime import datetime

from sqlalchemy import (Boolean, DateTime, Float, ForeignKey, Index, Integer, String, Text,
                        UniqueConstraint)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Общий базовый класс всех таблиц."""


# ==========================================================================
# Служебные таблицы: роли, пользователи, журнал, настройки
# ==========================================================================
class Role(Base):
    """Роль пользователя (уровень доступа)."""
    __tablename__ = "roles"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    code: Mapped[str] = mapped_column(String(32), unique=True)       # admin / analyst / viewer
    title: Mapped[str] = mapped_column(String(64))                   # название на русском
    description: Mapped[str] = mapped_column(String(255), default="")

    users: Mapped[list["User"]] = relationship(back_populates="role")


class User(Base):
    """Учётная запись пользователя системы."""
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    login: Mapped[str] = mapped_column(String(64), unique=True)
    password_hash: Mapped[str] = mapped_column(String(255))          # пароль хранится только в виде хеша
    full_name: Mapped[str] = mapped_column(String(128), default="")
    email: Mapped[str] = mapped_column(String(128), default="")
    position: Mapped[str] = mapped_column(String(128), default="")   # должность
    role_id: Mapped[int] = mapped_column(ForeignKey("roles.id"))
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    last_login_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    role: Mapped[Role] = relationship(back_populates="users")


class AuditLog(Base):
    """Журнал действий пользователей (кто, когда и что сделал)."""
    __tablename__ = "audit_log"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    action: Mapped[str] = mapped_column(String(64))                  # код действия: login, train, export ...
    details: Mapped[str] = mapped_column(Text, default="")
    ip_address: Mapped[str] = mapped_column(String(64), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now, index=True)

    user: Mapped[User] = relationship()


class Setting(Base):
    """Настройка системы в виде пары «ключ — значение»."""
    __tablename__ = "settings"

    key: Mapped[str] = mapped_column(String(64), primary_key=True)
    value: Mapped[str] = mapped_column(String(255), default="")
    title: Mapped[str] = mapped_column(String(255), default="")      # пояснение для панели настроек


# ==========================================================================
# Справочники и данные мониторинга
# ==========================================================================
class FederalDistrict(Base):
    """Федеральный округ."""
    __tablename__ = "federal_districts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(64), unique=True)

    regions: Mapped[list["Region"]] = relationship(back_populates="district")


class Region(Base):
    """Субъект Российской Федерации (идентификатор совпадает с кодом на сайте мониторинга)."""
    __tablename__ = "regions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    district_id: Mapped[int] = mapped_column(ForeignKey("federal_districts.id"))

    district: Mapped[FederalDistrict] = relationship(back_populates="regions")
    universities: Mapped[list["University"]] = relationship(back_populates="region")


class FounderGroup(Base):
    """Ведомственная принадлежность (Минобрнауки, Минздрав, частные организации и т. д.)."""
    __tablename__ = "founder_groups"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)

    universities: Mapped[list["University"]] = relationship(back_populates="founder_group")


class University(Base):
    """Образовательная организация высшего образования или её филиал."""
    __tablename__ = "universities"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)       # код организации на сайте мониторинга
    name: Mapped[str] = mapped_column(String(512), index=True)
    short_name: Mapped[str] = mapped_column(String(255), default="")
    is_branch: Mapped[bool] = mapped_column(Boolean, default=False)
    head_id: Mapped[int] = mapped_column(Integer, nullable=True)      # головная организация (для филиалов)
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id"))
    founder_group_id: Mapped[int] = mapped_column(ForeignKey("founder_groups.id"), nullable=True)
    city: Mapped[str] = mapped_column(String(128), default="")
    address: Mapped[str] = mapped_column(String(512), default="")
    website: Mapped[str] = mapped_column(String(255), default="")
    specifics: Mapped[str] = mapped_column(String(255), default="")  # отраслевая специфика
    longitude: Mapped[float] = mapped_column(Float, nullable=True)
    latitude: Mapped[float] = mapped_column(Float, nullable=True)
    last_year: Mapped[int] = mapped_column(Integer, default=0)        # последний год участия в мониторинге

    region: Mapped[Region] = relationship(back_populates="universities")
    founder_group: Mapped[FounderGroup] = relationship(back_populates="universities")
    values: Mapped[list["IndicatorValue"]] = relationship(back_populates="university")
    notes: Mapped[list["AnalystNote"]] = relationship(back_populates="university")


class Indicator(Base):
    """Справочник показателей мониторинга (основных и дополнительных)."""
    __tablename__ = "indicators"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    column: Mapped[str] = mapped_column(String(32), unique=True)     # имя столбца в наборе данных: e1, p48 ...
    code: Mapped[str] = mapped_column(String(16))                    # обозначение на сайте: E.1, 48 ...
    kind: Mapped[str] = mapped_column(String(16))                    # main — основной, extra — дополнительный
    section: Mapped[str] = mapped_column(String(128), default="")
    name: Mapped[str] = mapped_column(Text)
    unit: Mapped[str] = mapped_column(String(32), default="")


class IndicatorValue(Base):
    """Значение показателя организации за год (основная таблица фактов)."""
    __tablename__ = "indicator_values"
    # Составной первичный ключ без суррогатного идентификатора экономит место
    # и одновременно служит индексом для выборки «все показатели вуза за год».
    # Отдельный индекс по показателю не создаётся: он удвоил бы размер файла
    # базы, а выборки по показателю выполняются редко и на небольшом объёме
    __table_args__ = {"sqlite_with_rowid": False}

    university_id: Mapped[int] = mapped_column(ForeignKey("universities.id"), primary_key=True)
    year: Mapped[int] = mapped_column(Integer, primary_key=True)
    indicator_id: Mapped[int] = mapped_column(ForeignKey("indicators.id"), primary_key=True)
    value: Mapped[float] = mapped_column(Float, nullable=True)

    university: Mapped[University] = relationship(back_populates="values")
    indicator: Mapped[Indicator] = relationship()


class IndicatorMedian(Base):
    """Медианное значение показателя: по России, по субъекту РФ или по ведомству."""
    __tablename__ = "indicator_medians"
    __table_args__ = (UniqueConstraint("year", "indicator_id", "level", "region_id", "founder_group_id",
                                       name="uq_median_scope"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    year: Mapped[int] = mapped_column(Integer, index=True)
    indicator_id: Mapped[int] = mapped_column(ForeignKey("indicators.id"))
    level: Mapped[str] = mapped_column(String(16))                   # rf / region / founder
    region_id: Mapped[int] = mapped_column(ForeignKey("regions.id"), nullable=True)
    founder_group_id: Mapped[int] = mapped_column(ForeignKey("founder_groups.id"), nullable=True)
    value: Mapped[float] = mapped_column(Float, nullable=True)

    indicator: Mapped[Indicator] = relationship()


# ==========================================================================
# Результаты интеллектуального анализа
# ==========================================================================
class Dataset(Base):
    """Набор данных, загруженный в систему (собран с сайта или получен из файла)."""
    __tablename__ = "datasets"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    source: Mapped[str] = mapped_column(String(32))                  # site — сбор с сайта, file — загрузка файла
    location: Mapped[str] = mapped_column(String(512), default="")   # каталог или файл с данными
    years: Mapped[str] = mapped_column(String(64), default="")
    rows_count: Mapped[int] = mapped_column(Integer, default=0)
    description: Mapped[str] = mapped_column(Text, default="")
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    author: Mapped[User] = relationship()


class Experiment(Base):
    """Эксперимент — одно обучение модели с фиксированными гиперпараметрами."""
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    name: Mapped[str] = mapped_column(String(128))
    model_type: Mapped[str] = mapped_column(String(32))              # mlp / autoencoder / logreg / boosting ...
    params_json: Mapped[str] = mapped_column(Text, default="{}")     # гиперпараметры
    metrics_json: Mapped[str] = mapped_column(Text, default="{}")    # итоговые метрики качества
    status: Mapped[str] = mapped_column(String(16), default="new")   # new / running / done / failed
    message: Mapped[str] = mapped_column(Text, default="")           # текст ошибки или примечание
    model_file: Mapped[str] = mapped_column(String(255), default="") # путь к сохранённой модели
    is_active: Mapped[bool] = mapped_column(Boolean, default=False)  # используется ли для прогнозов
    dataset_id: Mapped[int] = mapped_column(ForeignKey("datasets.id"), nullable=True)
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)
    finished_at: Mapped[datetime] = mapped_column(DateTime, nullable=True)

    author: Mapped[User] = relationship()
    dataset: Mapped[Dataset] = relationship()
    epochs: Mapped[list["ExperimentEpoch"]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan", order_by="ExperimentEpoch.epoch")


class ExperimentEpoch(Base):
    """Метрики одной эпохи обучения (для построения кривых обучения)."""
    __tablename__ = "experiment_epochs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"), index=True)
    epoch: Mapped[int] = mapped_column(Integer)
    train_loss: Mapped[float] = mapped_column(Float, nullable=True)
    val_loss: Mapped[float] = mapped_column(Float, nullable=True)
    train_metric: Mapped[float] = mapped_column(Float, nullable=True)   # доля верных ответов на обучении
    val_metric: Mapped[float] = mapped_column(Float, nullable=True)     # доля верных ответов на валидации
    learning_rate: Mapped[float] = mapped_column(Float, nullable=True)

    experiment: Mapped[Experiment] = relationship(back_populates="epochs")


class Prediction(Base):
    """Прогноз класса результативности организации на следующий год мониторинга."""
    __tablename__ = "predictions"
    __table_args__ = (Index("ix_predictions_university", "university_id", "target_year"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    experiment_id: Mapped[int] = mapped_column(ForeignKey("experiments.id"))
    university_id: Mapped[int] = mapped_column(ForeignKey("universities.id"))
    base_year: Mapped[int] = mapped_column(Integer)                   # год исходных данных
    target_year: Mapped[int] = mapped_column(Integer)                 # год, на который строится прогноз
    predicted_class: Mapped[int] = mapped_column(Integer)
    probabilities_json: Mapped[str] = mapped_column(Text, default="[]")
    anomaly_score: Mapped[float] = mapped_column(Float, nullable=True)   # ошибка восстановления автокодировщика
    cluster: Mapped[int] = mapped_column(Integer, nullable=True)         # номер типологической группы
    created_by: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    experiment: Mapped[Experiment] = relationship()
    university: Mapped[University] = relationship()


class AnalystNote(Base):
    """Заметка аналитика к организации (комментарий к прогнозу, вывод, поручение)."""
    __tablename__ = "analyst_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    university_id: Mapped[int] = mapped_column(ForeignKey("universities.id"), index=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    text: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.now)

    university: Mapped[University] = relationship(back_populates="notes")
    user: Mapped[User] = relationship()
