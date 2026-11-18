# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модульные тесты разбора страниц сайта мониторинга (без обращения к сети).
"""

import pytest

from vuzmon.collect import parsers

INDEX_HTML = """
<table id='tregion'><tr><td>
 <div><div class='n_fo_name'>ЦФО</div>
   <p><a href='iam/2025/_vpo/material.php?type=2&id=10301'>г.Москва</a></p>
   <p><a href='iam/2025/_vpo/material.php?type=2&id=10501'>Белгородская область</a></p></div>
 <div><p><a href='iam/2025/_vpo/material.php?type=2&id=11305'>Запорожская область</a></p></div>
</td></tr></table>"""

REGION_HTML = """
<table><tr><td id=1943 icon='_2' city='г. Москва' coordinates='37.6, 55.7' class='inst'>
  <a href='inst.php?id=1943'>Частное образовательное учреждение "Университет".</a></td></tr>
<tr><td id=77 icon='_0' city='г. Рязань' coordinates='' class='inst'>
  <a href='inst.php?id=77'>Филиал университета в г. Рязани</a></td></tr></table>"""

INST_HTML = """
<table id='info'>
 <tr><td class='tt'>Наименование образовательной организации</td><td><b>Тестовый университет.</b></td></tr>
 <tr><td class='tt'>Регион,<br />адрес</td><td><a href='material.php?type=2&id=1'>г.Москва</a><br />
     <span id='post'>115432, г.Москва</span></td></tr>
 <tr><td class='tt'>Ведомственная принадлежность</td><td>Частные организации</td></tr>
 <tr><td class='tt'>Отраслевая специфика</td><td>образовательные организации, не имеющие специфики</td></tr>
</table>
<table id='result'><thead>
 <tr><td rowspan=2>Наименование показателя</td><td rowspan=2>2024</td><td rowspan=2>2025 | Динамика</td>
     <td colspan=3>Медианные значения по</td></tr>
 <tr><td>РФ</td><td>субъекту</td><td>ведомственной принадлежности</td></tr></thead>
 <tr><td class='n'>E.1. Образовательная деятельность<p>Средний балл ЕГЭ</p></td>
     <td>57,32</td><td><span>55,92|<sub>&#8211;2,4%</sub></span></td><td>62,80</td><td>66,10</td><td>53,19</td></tr>
 <tr><td class='n'>E.4. Финансово-экономическая деятельность<p>Доходы на одного НПР</p></td>
     <td>13 846,99</td><td>14 425,47| +4,2 %</td><td>4 241,60</td><td>5 044,20</td><td>3 599,00</td></tr>
</table>
<table id='analis_dop'>
 <tr><td>№ п/п</td><td>Наименование показателя</td><td>Единица измерения</td><td>Значение</td></tr>
 <tr><td>1</td><td>2</td><td>3</td><td>4</td></tr>
 <tr><td colspan=4>Образовательная деятельность</td></tr>
 <tr><td>1</td><td>Общая численность студентов</td><td>чел.</td><td>15 716</td></tr>
 <tr><td>в том числе: по очной форме обучения</td><td>чел.</td><td>2 233</td></tr>
 <tr><td colspan=4>Инфраструктура</td></tr>
 <tr><td>47</td><td>Наличие электронной библиотечной системы</td><td>да/нет</td><td>да</td></tr>
</table>
<table id='kont_by_otr'><tr><td></td><td>Доля</td><td>Контингент</td></tr>
 <tr><td>Науки об обществе</td><td>1,2%</td><td>4 100,5 чел.</td></tr></table>
<table id='analis_reg'><tr><td>38.00.00 - Экономика и управление</td><td>2 000,0</td><td>48,8%</td></tr>
 <tr><td>09.00.00 - Информатика</td><td>900,0</td><td>21,9%</td></tr></table>
<script>createChart([[1,2,3]]);</script>"""


@pytest.mark.parametrize("text, expected", [
    ("5 462,76", 5462.76), ("88,3 %", 88.3), ("575,0 чел.", 575.0), ("да", 1.0), ("нет", 0.0),
    ("–", None), ("", None), ("–2,1", -2.1), ("0,00", 0.0), ("показатель не оценивается", None),
    ("13\xa0846,99", 13846.99)])
def test_parse_number(text, expected):
    assert parsers.parse_number(text) == expected


def test_year_index_districts():
    regions = parsers.parse_year_index(INDEX_HTML)
    assert [(r.region_id, r.name, r.federal_district) for r in regions] == [
        (10301, "г.Москва", "ЦФО"), (10501, "Белгородская область", "ЦФО"),
        (11305, "Запорожская область", "Новые регионы")]


def test_year_index_without_labels_gives_unknown_district():
    """В материалах 2021–2023 годов подписей округов нет — округ остаётся пустым."""
    html = INDEX_HTML.replace("<div class='n_fo_name'>ЦФО</div>", "")
    assert {r.federal_district for r in parsers.parse_year_index(html)} == {""}


def test_region_page():
    head, branch = parsers.parse_region_page(REGION_HTML)
    assert (head.inst_id, head.is_branch, head.city) == (1943, False, "г. Москва")
    assert (head.longitude, head.latitude) == (37.6, 55.7)
    assert head.name.endswith('"Университет"')            # завершающая точка убирается
    assert branch.is_branch and branch.longitude is None


def test_institution_page():
    record = parsers.parse_institution_page(INST_HTML, 1943, 2025)
    assert record.info["name"] == "Тестовый университет"
    assert record.info["region"] == "г.Москва" and record.info["founder_group"] == "Частные организации"
    assert record.info["is_branch"] is False
    e1, e4 = record.main
    assert e1.code == "E.1" and e1.values == {2024: 57.32, 2025: 55.92}
    assert (e1.median_rf, e1.median_region, e1.median_founder) == (62.8, 66.1, 53.19)
    assert e4.values[2025] == 14425.47 and e4.title == "Финансово-экономическая деятельность"
    codes = {item.code: item for item in record.extra}
    assert set(codes) == {"1", "1.1", "47"}
    assert codes["1"].value == 15716 and codes["1.1"].name == "по очной форме обучения"
    assert codes["47"].value == 1.0 and codes["47"].section == "Инфраструктура"
    assert record.science_fields == {"Науки об обществе": 4100.5}
    assert (record.ugn_count, record.ugn_top_share) == (2, 48.8)


def test_institution_page_without_tables_is_not_an_error():
    record = parsers.parse_institution_page("<html><body>нет данных</body></html>", 1, 2025)
    assert record.main == [] and record.extra == [] and record.ugn_count == 0
