# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль parsers — извлечение структурированных данных из HTML-страниц
информационно-аналитических материалов мониторинга вузов.

Разбираются три вида страниц:
  1) оглавление года          — перечень субъектов РФ по федеральным округам;
  2) страница субъекта РФ     — перечень вузов и филиалов региона;
  3) страница организации     — общие сведения, основные показатели E.1–E.8
                                за пять лет, медианные значения и 56
                                дополнительных характеристик.
Источник слабо структурирован (таблицы свёрстаны для чтения человеком),
поэтому каждый разборщик устойчив к пропускам и отсутствующим блокам.
"""

import re
from dataclasses import dataclass, field

from bs4 import BeautifulSoup

# Признаки «значения нет»: прочерки разных видов и служебные фразы
_EMPTY_MARKS = {"", "-", "–", "—", "н/д", "нет данных", "показатель не оценивается"}
_YEAR_RE = re.compile(r"(20\d\d)")
_CODE_RE = re.compile(r"^(E\.\d+)\.?\s*(.*)$", re.S)
_UGN_RE = re.compile(r"^\d\d\.\d\d\.\d\d")


# --------------------------------------------------------------------------
# Структуры данных, которые возвращают разборщики
# --------------------------------------------------------------------------
@dataclass
class RegionRef:
    """Ссылка на страницу субъекта РФ в оглавлении года."""
    region_id: int
    name: str
    federal_district: str


@dataclass
class InstitutionRef:
    """Краткая запись об организации со страницы субъекта РФ."""
    inst_id: int
    name: str
    city: str
    is_branch: bool
    longitude: float = None
    latitude: float = None


@dataclass
class MainIndicator:
    """Основной показатель эффективности с пятилетней динамикой и медианами."""
    code: str                      # например, 'E.1'
    title: str                     # 'Образовательная деятельность'
    description: str               # полная формулировка показателя
    values: dict = field(default_factory=dict)   # {год: значение}
    median_rf: float = None        # медиана по России
    median_region: float = None    # медиана по субъекту РФ
    median_founder: float = None   # медиана по ведомственной принадлежности


@dataclass
class ExtraIndicator:
    """Дополнительная характеристика организации (раздел IV страницы)."""
    code: str        # '1', '1.1', '2', ... '56'
    section: str     # 'Образовательная деятельность', 'Кадровый потенциал', ...
    name: str
    unit: str
    value: float = None


@dataclass
class InstitutionRecord:
    """Полный набор сведений об организации за один год мониторинга."""
    inst_id: int
    year: int
    info: dict = field(default_factory=dict)
    main: list = field(default_factory=list)       # список MainIndicator
    extra: list = field(default_factory=list)      # список ExtraIndicator
    science_fields: dict = field(default_factory=dict)  # {отрасль наук: контингент}
    ugn_count: int = 0             # число реализуемых УГН(С)
    ugn_top_share: float = None    # доля крупнейшей УГН(С) в контингенте, %


# --------------------------------------------------------------------------
# Вспомогательные функции
# --------------------------------------------------------------------------
def clean_text(node) -> str:
    """Текст узла без лишних пробелов, переводов строк и неразрывных пробелов."""
    if node is None:
        return ""
    text = node.get_text(" ", strip=True) if hasattr(node, "get_text") else str(node)
    return re.sub(r"\s+", " ", text.replace("\xa0", " ")).strip()


def parse_number(text):
    """
    Преобразует число в российской записи в float.

    '5 462,76' -> 5462.76; '88,3 %' -> 88.3; '575,0 чел.' -> 575.0;
    'да' -> 1.0; 'нет' -> 0.0; прочерк или пустая строка -> None.
    """
    if text is None:
        return None
    value = re.sub(r"\s+", " ", str(text).replace("\xa0", " ")).strip().lower()
    if value in _EMPTY_MARKS:
        return None
    if value == "да":
        return 1.0
    if value == "нет":
        return 0.0
    # Минус на сайте набран типографским тире — приводим к обычному знаку
    value = value.replace("–", "-").replace("—", "-").replace("−", "-")
    match = re.search(r"-?\d[\d ]*(?:,\d+)?", value)
    if not match:
        return None
    try:
        return float(match.group(0).replace(" ", "").replace(",", "."))
    except ValueError:
        return None


def _make_soup(html: str) -> BeautifulSoup:
    return BeautifulSoup(html, "lxml")


# --------------------------------------------------------------------------
# 1. Оглавление года: субъекты РФ
# --------------------------------------------------------------------------
def parse_year_index(html: str) -> list:
    """Возвращает список RegionRef из страницы-оглавления года мониторинга."""
    soup = _make_soup(html)
    table = soup.find("table", id="tregion")
    if table is None:
        return []
    regions, seen = [], set()
    for link in table.find_all("a"):
        match = re.search(r"material\.php\?type=2&id=(\d+)", link.get("href", ""))
        if not match or int(match.group(1)) in seen:
            continue
        seen.add(int(match.group(1)))
        # Регионы сгруппированы в блоки <div>; подпись округа лежит внутри блока.
        # У блока новых регионов подписи нет, поэтому задаём её явно.
        block = link.find_parent("div")
        label = block.find("div", class_="n_fo_name") if block is not None else None
        district = clean_text(label) if label is not None else "Новые регионы"
        regions.append(RegionRef(int(match.group(1)), clean_text(link), district))
    return regions


# --------------------------------------------------------------------------
# 2. Страница субъекта РФ: перечень организаций
# --------------------------------------------------------------------------
def parse_region_page(html: str) -> list:
    """Возвращает список InstitutionRef со страницы субъекта РФ."""
    soup = _make_soup(html)
    result, seen = [], set()
    for cell in soup.find_all("td", class_="inst"):
        link = cell.find("a", href=re.compile(r"inst\.php\?id=\d+"))
        if link is None:
            continue
        inst_id = int(re.search(r"id=(\d+)", link["href"]).group(1))
        if inst_id in seen:
            continue
        seen.add(inst_id)
        longitude = latitude = None
        # Координаты записаны через запятую с десятичной точкой: '40.549260, 64.529183'
        try:
            longitude, latitude = (float(part) for part in cell.get("coordinates", "").split(","))
        except ValueError:
            longitude = latitude = None
        # Значок '_2' на сайте означает головную организацию, '_0' — филиал
        is_branch = cell.get("icon", "") != "_2"
        result.append(InstitutionRef(inst_id, clean_text(link).rstrip("."),
                                     cell.get("city", "").strip(), is_branch,
                                     longitude, latitude))
    return result


# --------------------------------------------------------------------------
# 3. Страница организации
# --------------------------------------------------------------------------
def _parse_info(soup: BeautifulSoup) -> dict:
    """Блок «Общие сведения»: название, регион, ведомство, специфика и т. д."""
    info = {}
    table = soup.find("table", id="info")
    if table is None:
        return info
    head_part = False   # после строки «Сведения о головной организации» идут её данные
    for row in table.find_all("tr"):
        cells = row.find_all("td")
        if len(cells) == 1 and "головной" in clean_text(cells[0]).lower():
            head_part = True
            continue
        if len(cells) != 2:
            continue
        label = clean_text(cells[0]).lower()
        value_cell = cells[1]
        if head_part:
            if label.startswith("наименование"):
                info["head_name"] = clean_text(value_cell)
                link = value_cell.find("a", href=re.compile(r"inst\.php\?id=\d+"))
                if link is not None:
                    info["head_id"] = int(re.search(r"id=(\d+)", link["href"]).group(1))
            elif label.startswith("регион"):
                info["head_region"] = clean_text(value_cell)
        elif label.startswith("наименование"):
            info["name"] = clean_text(value_cell).rstrip(".")
        elif label.startswith("регион"):
            link = value_cell.find("a")
            info["region"] = clean_text(link) if link is not None else ""
            post = value_cell.find(id="post")
            info["address"] = clean_text(post)
        elif label.startswith("ведомственная"):
            info["founder_group"] = clean_text(value_cell)
        elif label.startswith("web"):
            info["website"] = clean_text(value_cell)
        elif label.startswith("учредител"):
            info["founders"] = "; ".join(clean_text(p) for p in value_cell.find_all("p")) \
                or clean_text(value_cell)
        elif label.startswith("отраслевая"):
            info["specifics"] = clean_text(value_cell)
    info["is_branch"] = head_part
    return info


def _parse_main(soup: BeautifulSoup) -> list:
    """Таблица основных показателей: значения за 5 лет и три медианы."""
    table = soup.find("table", id="result")
    if table is None:
        return []
    # Годы берём из шапки таблицы (в ячейке последнего года стоит «2025 | Динамика»)
    years = []
    head = table.find("thead")
    if head is not None:
        for cell in head.find_all("td"):
            match = _YEAR_RE.match(clean_text(cell))
            if match:
                years.append(int(match.group(1)))
    if not years:
        return []
    indicators = []
    for row in table.find_all("tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 1 + len(years) + 3:
            continue
        name_cell = cells[0]
        description = clean_text(name_cell.find("p"))
        full_title = clean_text(name_cell)
        if description and full_title.endswith(description):
            full_title = full_title[: -len(description)].strip()
        match = _CODE_RE.match(full_title)
        if not match:
            continue
        indicator = MainIndicator(code=match.group(1), title=match.group(2).strip(),
                                  description=description)
        for year, cell in zip(years, cells[1:1 + len(years)]):
            # В ячейке текущего года после черты указана динамика: '65,36| –2,1 %'
            raw = clean_text(cell).split("|")[0]
            indicator.values[year] = parse_number(raw)
        medians = [parse_number(clean_text(c)) for c in cells[1 + len(years):4 + len(years)]]
        indicator.median_rf, indicator.median_region, indicator.median_founder = medians
        indicators.append(indicator)
    return indicators


def _parse_extra(soup: BeautifulSoup) -> list:
    """Таблица дополнительных характеристик (56 пронумерованных показателей)."""
    table = soup.find("table", id="analis_dop")
    if table is None:
        return []
    result = []
    section, parent_code, sub_index = "", "", 0
    for row in table.find_all("tr"):
        cells = [clean_text(c) for c in row.find_all(["td", "th"], recursive=False)]
        if len(cells) == 1 and cells[0]:
            section = cells[0]                       # строка-заголовок раздела
        elif len(cells) == 4 and cells[0].isdigit() and cells[1] != "2":
            parent_code, sub_index = cells[0], 0      # обычный пронумерованный показатель
            result.append(ExtraIndicator(parent_code, section, cells[1], cells[2],
                                         parse_number(cells[3])))
        elif len(cells) == 3 and parent_code:
            # Строка-уточнение («в том числе: по очной форме обучения»)
            sub_index += 1
            name = re.sub(r"^в том числе:\s*", "", cells[0])
            result.append(ExtraIndicator(f"{parent_code}.{sub_index}", section, name,
                                         cells[1], parse_number(cells[2])))
    return result


def _parse_science_fields(soup: BeautifulSoup) -> dict:
    """Распределение приведённого контингента по девяти отраслям наук."""
    table = soup.find("table", id="kont_by_otr")
    fields = {}
    if table is None:
        return fields
    for row in table.find_all("tr"):
        cells = [clean_text(c) for c in row.find_all("td")]
        if len(cells) >= 3 and cells[0]:
            fields[cells[0]] = parse_number(cells[2]) or 0.0
    return fields


def _parse_ugn(soup: BeautifulSoup):
    """Число реализуемых УГН(С) и доля крупнейшей из них в контингенте."""
    table = soup.find("table", id="analis_reg")
    if table is None:
        return 0, None
    shares = []
    for row in table.find_all("tr"):
        cells = [clean_text(c) for c in row.find_all("td")]
        if len(cells) >= 3 and _UGN_RE.match(cells[0]):
            shares.append(parse_number(cells[2]) or 0.0)
    return len(shares), (max(shares) if shares else None)


def parse_institution_page(html: str, inst_id: int, year: int) -> InstitutionRecord:
    """Разбирает страницу организации и возвращает InstitutionRecord."""
    soup = _make_soup(html)
    # Скрипты содержат копии данных для графиков и только мешают разбору
    for script in soup.find_all("script"):
        script.decompose()
    record = InstitutionRecord(inst_id=inst_id, year=year)
    record.info = _parse_info(soup)
    record.main = _parse_main(soup)
    record.extra = _parse_extra(soup)
    record.science_fields = _parse_science_fields(soup)
    record.ugn_count, record.ugn_top_share = _parse_ugn(soup)
    return record
