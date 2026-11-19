# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль crawler — обход сайта мониторинга и формирование набора данных.

Порядок обхода для каждого года:
    оглавление года -> страницы субъектов РФ -> страницы организаций.
Страницы организаций скачиваются и разбираются в несколько потоков.
Результат сводится в таблицы pandas и сохраняется в CSV-файлы, которые
затем используются и для обучения моделей, и для наполнения базы данных.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from vuzmon import config
from vuzmon.collect import parsers
from vuzmon.collect.http_client import CachedHttpClient, DownloadError

logger = logging.getLogger(__name__)

# Короткие имена столбцов для отраслей наук (по первым буквам названия на сайте)
SCIENCE_FIELD_COLUMNS = {
    "математ": "sf_math",   # математические и естественные науки
    "инженер": "sf_eng",    # инженерное дело, технологии и технические науки
    "здраво": "sf_med",     # здравоохранение и медицинские науки
    "сельск": "sf_agro",    # сельское хозяйство и сельскохозяйственные науки
    "науки об": "sf_soc",   # науки об обществе
    "образов": "sf_edu",    # образование и педагогические науки
    "гуманит": "sf_hum",    # гуманитарные науки
    "искусст": "sf_art",    # искусство и культура
    "оборона": "sf_def",    # оборона и безопасность государства, военные науки
}


def extra_column(code: str) -> str:
    """Имя столбца дополнительной характеристики: '1.2' -> 'p1_2', '48' -> 'p48'."""
    return "p" + code.replace(".", "_")


@dataclass
class CrawlResult:
    """Итог обхода: разобранные записи и протокол ошибок."""
    records: list = field(default_factory=list)      # список (InstitutionRecord, InstitutionRef, RegionRef)
    errors: list = field(default_factory=list)       # список строк с описанием сбоев


class MonitoringCrawler:
    """Обходит сайт мониторинга и собирает сведения об организациях."""

    def __init__(self, client: CachedHttpClient, base_url: str = config.MONITORING_BASE_URL,
                 workers: int = 8, progress=None):
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.workers = workers
        # progress(done, total, message) — необязательная функция для отображения хода работ
        self.progress = progress or (lambda done, total, message: None)

    # ----------------------------------------------------------- адреса страниц
    def _index_url(self, year: int) -> str:
        return f"{self.base_url}/?m=vpo&year={year}"

    def _region_url(self, year: int, region_id: int) -> str:
        return f"{self.base_url}/iam/{year}/_vpo/material.php?type=2&id={region_id}"

    def _inst_url(self, year: int, inst_id: int) -> str:
        return f"{self.base_url}/iam/{year}/_vpo/inst.php?id={inst_id}"

    # ------------------------------------------------------------ шаги обхода
    def fetch_regions(self, year: int) -> list:
        """Перечень субъектов РФ, представленных в мониторинге указанного года."""
        html = self.client.get(self._index_url(year), f"{year}/index", must_contain="tregion")
        return parsers.parse_year_index(html)

    def fetch_region_institutions(self, year: int, region) -> list:
        """Перечень организаций субъекта РФ."""
        html = self.client.get(self._region_url(year, region.region_id),
                               f"{year}/region_{region.region_id}", must_contain="instheader")
        return parsers.parse_region_page(html)

    def fetch_institution(self, year: int, ref) -> parsers.InstitutionRecord:
        """Страница организации, разобранная в InstitutionRecord."""
        html = self.client.get(self._inst_url(year, ref.inst_id),
                               f"{year}/inst_{ref.inst_id}", must_contain="id='info'")
        return parsers.parse_institution_page(html, ref.inst_id, year)

    def crawl_year(self, year: int, limit: int = None) -> CrawlResult:
        """
        Полный обход одного года мониторинга.

        limit — ограничение числа организаций (используется в тестах и для
        быстрой проверки работоспособности сборщика).
        """
        result = CrawlResult()
        regions = self.fetch_regions(year)
        logger.info("Год %s: найдено субъектов РФ — %s", year, len(regions))

        # 1) Страницы субъектов скачиваем параллельно и собираем общий список организаций
        tasks = []   # пары (ссылка на организацию, регион)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.fetch_region_institutions, year, region): region
                       for region in regions}
            for future in as_completed(futures):
                region = futures[future]
                try:
                    tasks.extend((ref, region) for ref in future.result())
                except DownloadError as error:
                    result.errors.append(f"{year}: регион {region.name}: {error}")
        tasks.sort(key=lambda pair: pair[0].inst_id)
        if limit:
            tasks = tasks[:limit]
        logger.info("Год %s: организаций к обработке — %s", year, len(tasks))

        # 2) Страницы организаций: скачивание и разбор в пуле потоков
        done = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = {pool.submit(self.fetch_institution, year, ref): (ref, region)
                       for ref, region in tasks}
            for future in as_completed(futures):
                ref, region = futures[future]
                done += 1
                try:
                    result.records.append((future.result(), ref, region))
                except DownloadError as error:
                    result.errors.append(f"{year}: организация {ref.inst_id}: {error}")
                except Exception as error:  # ошибка разбора не должна останавливать сбор
                    result.errors.append(f"{year}: организация {ref.inst_id}: разбор: {error!r}")
                if done % 25 == 0 or done == len(tasks):
                    self.progress(done, len(tasks), f"{year}: обработано {done} из {len(tasks)}")
        return result

    def crawl(self, years, limit: int = None) -> CrawlResult:
        """Обход нескольких лет мониторинга с объединением результатов."""
        total = CrawlResult()
        for year in years:
            part = self.crawl_year(year, limit=limit)
            total.records.extend(part.records)
            total.errors.extend(part.errors)
        return total


# --------------------------------------------------------------------------
# Преобразование записей в таблицы и сохранение
# --------------------------------------------------------------------------
UNKNOWN_DISTRICT = "Не указан"


def normalize_regions(institutions: pd.DataFrame) -> pd.DataFrame:
    """
    Приводит сведения о регионах к единому виду во всех годах.

    * Федеральный округ: в материалах части лет он не указан, поэтому берётся
      наиболее частое известное значение для того же кода региона.
    * Название региона: на сайте встречаются варианты написания («г. Москва»
      и «г.Москва»); используется написание последнего года.
    """
    frame = institutions.copy()
    frame["federal_district"] = frame["federal_district"].fillna("").replace(UNKNOWN_DISTRICT, "")
    known = frame[frame["federal_district"] != ""]
    if not known.empty:
        by_region = known.groupby("region_id")["federal_district"].agg(lambda s: s.mode().iloc[0])
        frame["federal_district"] = frame["region_id"].map(by_region).fillna(frame["federal_district"])
    frame["federal_district"] = frame["federal_district"].replace("", UNKNOWN_DISTRICT)
    latest_names = frame.sort_values("year").drop_duplicates("region_id", keep="last") \
        .set_index("region_id")["region"]
    frame["region"] = frame["region_id"].map(latest_names)
    return frame


def records_to_frames(records: list) -> dict:
    """
    Сводит разобранные записи в набор таблиц pandas:
      institutions — паспорт организации за год;
      main         — основные показатели (строка = организация, страница года, год значения);
      medians      — медианы по РФ, региону и ведомству;
      extra        — дополнительные характеристики в «широком» виде;
      dictionary   — справочник показателей.
    """
    inst_rows, main_rows, median_rows, extra_rows = [], [], [], []
    dictionary = {}

    for record, ref, region in records:
        info = record.info
        row = {
            "year": record.year,
            "inst_id": record.inst_id,
            "name": info.get("name") or ref.name,
            "is_branch": int(ref.is_branch or info.get("is_branch", False)),
            "head_id": info.get("head_id"),
            "head_name": info.get("head_name", ""),
            "region_id": region.region_id,
            "region": region.name,
            "federal_district": region.federal_district,
            "city": ref.city,
            "longitude": ref.longitude,
            "latitude": ref.latitude,
            "address": info.get("address", ""),
            "founder_group": info.get("founder_group", ""),
            "founders": info.get("founders", ""),
            "website": info.get("website", ""),
            "specifics": info.get("specifics", ""),
            "ugn_count": record.ugn_count,
            "ugn_top_share": record.ugn_top_share,
        }
        # Контингент по отраслям наук раскладываем по коротким столбцам
        for column in SCIENCE_FIELD_COLUMNS.values():
            row[column] = 0.0
        for field_name, value in record.science_fields.items():
            for prefix, column in SCIENCE_FIELD_COLUMNS.items():
                if field_name.lower().startswith(prefix):
                    row[column] = value
        inst_rows.append(row)

        # Основные показатели: по одной строке на каждый год значения
        by_year = {}
        for indicator in record.main:
            column = config.MAIN_INDICATORS.get(indicator.code)
            if column is None:
                continue
            for value_year, value in indicator.values.items():
                by_year.setdefault(value_year, {})[column] = value
            median_rows.append({
                "year": record.year, "inst_id": record.inst_id, "code": column,
                "median_rf": indicator.median_rf, "median_region": indicator.median_region,
                "median_founder": indicator.median_founder,
            })
            dictionary[column] = {"column": column, "code": indicator.code, "section": "Основные показатели",
                                  "name": f"{indicator.title}. {indicator.description}".strip(),
                                  "unit": ""}
        for value_year, values in by_year.items():
            main_rows.append({"inst_id": record.inst_id, "page_year": record.year,
                              "value_year": value_year, **values})

        # Дополнительные характеристики: одна «широкая» строка на организацию и год
        extra_row = {"year": record.year, "inst_id": record.inst_id}
        for indicator in record.extra:
            column = extra_column(indicator.code)
            extra_row[column] = indicator.value
            dictionary[column] = {"column": column, "code": indicator.code, "section": indicator.section,
                                  "name": indicator.name, "unit": indicator.unit}
        extra_rows.append(extra_row)

    main_columns = ["inst_id", "page_year", "value_year"] + list(config.MAIN_INDICATORS.values())
    frames = {
        "institutions": normalize_regions(pd.DataFrame(inst_rows)) if inst_rows else pd.DataFrame(inst_rows),
        "main": pd.DataFrame(main_rows).reindex(columns=main_columns),
        "medians": pd.DataFrame(median_rows),
        "extra": pd.DataFrame(extra_rows),
        "dictionary": pd.DataFrame(list(dictionary.values())),
    }
    # Единый порядок строк делает файлы воспроизводимыми от запуска к запуску
    sort_keys = {"institutions": ["year", "inst_id"], "main": ["inst_id", "page_year", "value_year"],
                 "medians": ["year", "inst_id", "code"], "extra": ["year", "inst_id"],
                 "dictionary": ["column"]}
    for name, keys in sort_keys.items():
        if not frames[name].empty:
            frames[name] = frames[name].sort_values(keys).reset_index(drop=True)
    return frames


# Имена файлов набора данных
DATASET_FILES = {
    "institutions": "institutions.csv",
    "main": "main_indicators.csv",
    "medians": "medians.csv",
    "extra": "extra_indicators.csv",
    "dictionary": "indicator_dictionary.csv",
}


def save_frames(frames: dict, out_dir=config.MONITORING_DIR) -> dict:
    """Сохраняет таблицы в CSV (UTF-8, разделитель — запятая). Возвращает пути файлов."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, filename in DATASET_FILES.items():
        path = out_dir / filename
        frames[name].to_csv(path, index=False, encoding="utf-8")
        paths[name] = path
    return paths


def load_frames(data_dir=config.MONITORING_DIR) -> dict:
    """Читает ранее сохранённый набор данных из каталога CSV-файлов."""
    data_dir = Path(data_dir)
    return {name: pd.read_csv(data_dir / filename, encoding="utf-8")
            for name, filename in DATASET_FILES.items()}


def load_frames_from_url(base_url: str = config.DATASET_URL) -> dict:
    """
    Загружает набор данных напрямую по ссылке: base_url — адрес каталога, в
    котором лежат CSV-файлы набора (по умолчанию — каталог data/monitoring
    в публичном репозитории проекта), либо адрес ZIP-архива с этими файлами.

    Скачивание выполняет модуль remote: с таймаутом, ограничением размера и
    понятными сообщениями об ошибках (исключение ValueError).
    """
    from vuzmon.collect.remote import load_frames_from_url as load   # модуль remote сам использует crawler
    return load(base_url)
