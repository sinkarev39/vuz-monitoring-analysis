# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Проверка установки. Сценарий обращается к запущенному приложению так же, как
браузер, проходит основные сценарии работы и печатает протокол: для каждого
сценария — ожидаемый и фактический результат.

    python main.py                                   — в первом окне: запуск системы
    python check_install.py                          — во втором окне: проверка http://127.0.0.1:5000
    python check_install.py --url http://127.0.0.1:8080
    python check_install.py --collect                — дополнительно получить данные с сайта мониторинга
    python check_install.py --wait 120               — ждать запуска приложения до 120 секунд
    python check_install.py --report protocol.md     — сохранить протокол в файл

Проверяются: вход под тремя ролями и состав меню, запрет чужих разделов,
построение прогноза, выгрузка документов Word и Excel (каждый документ
открывается библиотекой и проверяется его содержимое), сравнение организаций,
проверка настроек, защитные заголовки ответов и приостановка входа после
нескольких неверных паролей. Получение данных из внешних источников требует
доступа к Интернету, поэтому включается отдельным ключом: с сайта мониторинга
запрашиваются сведения о трёх организациях за один год, а опубликованный набор
данных загружается по ссылке из репозитория проекта. Оба набора сохраняются
в перечне наборов, а рабочий набор данных остаётся прежним.

Код возврата: 0 — все сценарии выполнены, 1 — есть расхождения.
"""

import argparse
import io
import platform
import re
import sys
import time

import requests
from docx import Document
from openpyxl import load_workbook

from vuzmon import config

# Тестовые учётные записи (те же, что в README.md): роль -> (логин, пароль, название роли, пунктов меню)
ACCOUNTS = {
    "admin": ("admin", "Admin#2026", "Администратор", 12),
    "analyst": ("analyst", "Analyst#2026", "Аналитик", 11),
    "viewer": ("viewer", "Viewer#2026", "Руководитель", 9),
}
# Разделы с ограниченным доступом: адрес -> роли, которым раздел открыт
RESTRICTED = {"/data/": ("admin", "analyst"), "/models/": ("admin", "analyst"), "/admin/users": ("admin",)}

COLLECT_LIMIT = 3          # сколько организаций запрашивать при проверке сбора данных
COLLECT_TIMEOUT = 300      # предельное время ожидания сбора, с
LINK_TIMEOUT = 600         # предельное время ожидания загрузки набора по ссылке, с
LOGIN_ATTEMPTS = 5         # после стольких неверных паролей подряд вход приостанавливается


class Visitor:
    """Посетитель сайта: хранит cookie сессии и подставляет в формы токен защиты."""

    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.http = requests.Session()
        self.http.trust_env = False      # приложение локальное, системный прокси не нужен
        self.token = ""

    def get(self, path: str, follow: bool = True) -> requests.Response:
        response = self.http.get(self.base_url + path, allow_redirects=follow, timeout=120)
        self._remember_token(response)
        return response

    def post(self, path: str, data: dict = None) -> requests.Response:
        form = dict(data or {}, _csrf=self.token)
        response = self.http.post(self.base_url + path, data=form, timeout=300)
        self._remember_token(response)
        return response

    def _remember_token(self, response: requests.Response) -> None:
        found = re.search(r'name="_csrf" value="([0-9a-f]+)"', response.text) if "html" in \
            response.headers.get("Content-Type", "") else None
        if found:
            self.token = found.group(1)

    def login(self, role: str, password: str = None) -> requests.Response:
        login, real_password = ACCOUNTS[role][:2]
        self.get("/login")
        return self.post("/login", {"login": login, "password": password or real_password})


def flashes(html: str) -> str:
    """Сообщения, показанные пользователю на странице (без разметки)."""
    found = re.findall(r'<div class="flash [a-z]+">(.*?)</div>', html, flags=re.S)
    return " ".join(re.sub(r"<[^>]+>", "", text).strip() for text in found)


def menu_titles(html: str) -> list:
    """Названия пунктов главного меню на странице."""
    menu = re.search(r'<ul class="menu">(.*?)</ul>', html, flags=re.S)
    return re.findall(r"</span>([^<]+)</a>", menu.group(1)) if menu else []


class Protocol:
    """Протокол проверки: строки «сценарий — ожидалось — получено — итог»."""

    def __init__(self):
        self.rows = []

    def add(self, title: str, expected: str, actual: str, passed: bool) -> None:
        self.rows.append((title, expected, actual, passed))
        print(f"[{'выполнен' if passed else 'РАСХОЖДЕНИЕ'}] {title}\n    ожидалось: {expected}\n    получено:  {actual}")

    def run(self, title: str, expected: str, check) -> None:
        """Выполняет проверку; check() возвращает (фактический результат, признак совпадения)."""
        try:
            actual, passed = check()
        except Exception as error:   # сбой проверки — это тоже результат, протокол должен быть полным
            actual, passed = f"ошибка: {type(error).__name__}: {error}", False
        self.add(title, expected, actual, passed)

    @property
    def failed(self) -> int:
        return sum(1 for row in self.rows if not row[3])

    def markdown(self, header: list) -> str:
        lines = ["# Протокол проверки установки", ""] + [f"- {line}" for line in header] + [
            "", "| № | Сценарий | Ожидаемый результат | Фактический результат | Итог |", "|---|---|---|---|---|"]
        for number, (title, expected, actual, passed) in enumerate(self.rows, start=1):
            lines.append(f"| {number} | {title} | {expected} | {actual} | {'выполнен' if passed else 'расхождение'} |")
        lines += ["", f"Сценариев: {len(self.rows)}, выполнено: {len(self.rows) - self.failed}, "
                      f"расхождений: {self.failed}."]
        return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
# Сценарии
# --------------------------------------------------------------------------
def check_entrance(protocol: Protocol, url: str) -> None:
    guest = Visitor(url)

    def login_page():
        response = guest.get("/login")
        has_form = 'name="login"' in response.text and 'name="password"' in response.text
        return f"код {response.status_code}, форма входа {'показана' if has_form else 'не найдена'}", \
            response.status_code == 200 and has_form
    protocol.run("Страница входа открывается без авторизации", "код 200, форма входа показана", login_page)

    def closed_page():
        response = guest.get("/forecast", follow=False)
        target = response.headers.get("Location", "")
        return f"код {response.status_code}, переход на {target.split('?')[0] or '—'}", \
            response.status_code == 302 and target.startswith("/login")
    protocol.run("Страница «Прогноз» без входа в систему", "код 302, переход на /login", closed_page)

    def wrong_password():
        response = guest.login("admin", password="wrong-password")
        message = flashes(response.text)
        return f"вход не выполнен, сообщение «{message}»", "Неверный логин или пароль" in message
    protocol.run("Вход с неверным паролем", "вход не выполнен, сообщение «Неверный логин или пароль.»",
                 wrong_password)


def check_role(protocol: Protocol, url: str, role: str) -> Visitor:
    """Вход под ролью, состав меню и доступ к разделам. Возвращает посетителя, вошедшего в систему."""
    title, items = ACCOUNTS[role][2], ACCOUNTS[role][3]
    visitor = Visitor(url)

    def enter():
        response = visitor.login(role)
        shown = re.search(r'<div class="role">([^<]+)</div>', response.text)
        menu = menu_titles(response.text)
        actual = f"открыта страница «{'Обзор' if '<h1>Обзор' in response.text else '?'}», роль " \
                 f"«{shown.group(1) if shown else '—'}», пунктов меню: {len(menu)}"
        return actual, bool(shown) and shown.group(1) == title and len(menu) == items and "<h1>Обзор" in response.text
    protocol.run(f"Вход под ролью «{title.lower()}»",
                 f"открыта страница «Обзор», роль «{title}», пунктов меню: {items}", enter)

    def sections():
        codes = {path: visitor.get(path).status_code for path in RESTRICTED}
        wanted = {path: 200 if role in roles else 403 for path, roles in RESTRICTED.items()}
        return ", ".join(f"{path} — {code}" for path, code in codes.items()), codes == wanted
    expected = ", ".join(f"{path} — {200 if role in roles else 403}" for path, roles in RESTRICTED.items())
    protocol.run(f"Доступ к разделам «Данные», «Модели», «Администрирование»: {title.lower()}", expected, sections)
    return visitor


def check_forecast(protocol: Protocol, analyst: Visitor, viewer: Visitor) -> None:
    def forbidden():
        code = viewer.post("/forecast/build").status_code
        return f"код {code}", code == 403
    protocol.run("Построение прогноза руководителем (действие ему запрещено)", "код 403", forbidden)

    def build():
        response = analyst.post("/forecast/build")
        message = flashes(response.text)
        rows = len(re.findall(r'href="/universities/\d+"', response.text))
        return f"сообщение «{message}», строк на первой странице таблицы: {rows}", \
            "Прогноз построен для" in message and rows > 0
    protocol.run("Построение прогноза аналитиком",
                 "сообщение «Прогноз построен для N организаций по данным года Y.», таблица прогноза заполнена", build)


def check_documents(protocol: Protocol, viewer: Visitor, admin: Visitor) -> None:
    def workbook(visitor: Visitor, path: str):
        response = visitor.get(path)
        book = load_workbook(io.BytesIO(response.content))       # ошибка здесь означает повреждённый файл
        return response, book.active

    def forecast_xlsx():
        response, sheet = workbook(viewer, "/reports/forecast.xlsx")
        return f"код {response.status_code}, файл Excel открыт, строк в таблице: {sheet.max_row}", \
            response.status_code == 200 and sheet.max_row > 100
    protocol.run("Выгрузка таблицы прогноза в Excel (руководитель)",
                 "код 200, файл Excel открыт, в таблице более 100 строк", forecast_xlsx)

    def word(path: str):
        response = viewer.get(path)
        document = Document(io.BytesIO(response.content))        # ошибка здесь означает повреждённый файл
        text = [paragraph.text for paragraph in document.paragraphs if paragraph.text.strip()]
        return response, document, text

    def university_docx():
        page = viewer.get("/forecast").text
        university_id = re.search(r'href="/universities/(\d+)"', page).group(1)
        response, document, text = word(f"/reports/university/{university_id}.docx")
        return f"код {response.status_code}, документ Word открыт: абзацев {len(text)}, таблиц " \
               f"{len(document.tables)}", response.status_code == 200 and len(text) > 3 and len(document.tables) > 0
    protocol.run("Справка по организации в Word (руководитель)",
                 "код 200, документ Word открыт, содержит текст и таблицу показателей", university_docx)

    def summary_docx():
        response, document, text = word("/reports/summary.docx")
        return f"код {response.status_code}, документ Word открыт: абзацев {len(text)}, таблиц " \
               f"{len(document.tables)}", response.status_code == 200 and len(text) > 3 and len(document.tables) > 0
    protocol.run("Сводная аналитическая записка в Word (руководитель)",
                 "код 200, документ Word открыт, содержит текст и таблицы", summary_docx)

    def audit_xlsx():
        response, sheet = workbook(admin, "/reports/audit.xlsx")
        cells = " ".join(str(value) for row in sheet.iter_rows(values_only=True) for value in row if value)
        return f"код {response.status_code}, файл Excel открыт, строк: {sheet.max_row}, запись о построении " \
               f"прогноза {'есть' if 'forecast_build' in cells or 'Построен прогноз' in cells else 'не найдена'}", \
            response.status_code == 200 and ("forecast_build" in cells or "Построен прогноз" in cells)
    protocol.run("Журнал действий в Excel (администратор)",
                 "код 200, файл Excel открыт, есть запись о построении прогноза", audit_xlsx)

    def audit_closed():
        code = viewer.get("/reports/audit.xlsx").status_code
        return f"код {code}", code == 403
    protocol.run("Журнал действий недоступен руководителю", "код 403", audit_closed)


def check_protection(protocol: Protocol, url: str, viewer: Visitor, admin: Visitor) -> None:
    """Сравнение с повторным выбором, проверка настроек, защитные заголовки, приостановка входа."""
    def compare_twice():
        university_id = re.search(r'href="/universities/(\d+)"', viewer.get("/forecast").text).group(1)
        response = viewer.get(f"/compare?id={university_id}&id={university_id}")
        shown = response.text.count(">убрать<")
        return f"код {response.status_code}, организаций в сравнении: {shown}", \
            response.status_code == 200 and shown == 1
    protocol.run("Сравнение: одна и та же организация выбрана дважды",
                 "код 200, организаций в сравнении: 1", compare_twice)

    def wrong_setting():
        admin.get("/admin/settings")
        message = flashes(admin.post("/admin/settings", {"records_per_page": "много"}).text)
        code = viewer.get("/forecast").status_code
        return f"сообщение «{message}», страница «Прогноз» — код {code}", \
            "Настройки не сохранены" in message and "целое число от 5 до 200" in message and code == 200
    protocol.run("Настройки: текст вместо числа строк на странице (администратор)",
                 "сообщение о допустимых значениях («целое число от 5 до 200»), настройки не сохранены, "
                 "страница «Прогноз» — код 200", wrong_setting)

    def headers():
        got = Visitor(url).get("/login").headers
        found = {"X-Frame-Options": got.get("X-Frame-Options", "нет"),
                 "X-Content-Type-Options": got.get("X-Content-Type-Options", "нет"),
                 "Cache-Control": got.get("Cache-Control", "нет"),
                 "Content-Security-Policy": "есть" if "default-src 'self'" in got.get("Content-Security-Policy", "")
                 else "нет"}
        wanted = {"X-Frame-Options": "DENY", "X-Content-Type-Options": "nosniff", "Cache-Control": "no-store",
                  "Content-Security-Policy": "есть"}
        return ", ".join(f"{name}: {value}" for name, value in found.items()), found == wanted
    protocol.run("Защитные заголовки ответа сервера",
                 "X-Frame-Options: DENY, X-Content-Type-Options: nosniff, Cache-Control: no-store, "
                 "Content-Security-Policy: есть", headers)

    def lockout():
        # Логин заведомо несуществующий и новый при каждой проверке: учётные записи системы не затрагиваются
        guest, name = Visitor(url), f"check{int(time.time())}"
        guest.get("/login")
        codes = [guest.post("/login", {"login": name, "password": "wrong-password"}).status_code
                 for _ in range(LOGIN_ATTEMPTS)]
        response = guest.post("/login", {"login": name, "password": "wrong-password"})
        message = flashes(response.text)
        return f"{LOGIN_ATTEMPTS} попыток — коды {sorted(set(codes))}, следующая — код {response.status_code}, " \
               f"сообщение «{message}»", set(codes) == {200} and response.status_code == 429 \
            and "Слишком много неудачных попыток входа" in message
    protocol.run(f"Приостановка входа после {LOGIN_ATTEMPTS} неверных паролей подряд",
                 f"{LOGIN_ATTEMPTS} попыток — коды [200], следующая — код 429, сообщение «Слишком много неудачных "
                 f"попыток входа. Повторите через N мин.»", lockout)


def working_dataset(visitor: Visitor) -> str:
    """Номер и название рабочего набора данных со страницы «Данные»."""
    found = re.search(r'<a href="/data/(\d+)">([^<]+)</a>\s*<span class="badge ok">рабочий</span>',
                      visitor.get("/data/").text)
    return f"№{found.group(1)} «{found.group(2)}»" if found else "не определён"


def check_collect(protocol: Protocol, analyst: Visitor) -> None:
    year = config.DEFAULT_YEARS[-1]
    before = working_dataset(analyst)

    def collect():
        # Отметка «сделать рабочим» поставлена намеренно: пробный набор за один год рабочим стать не должен
        analyst.post("/data/collect", {"years": str(year), "limit": str(COLLECT_LIMIT), "load_to_db": "1"})
        started = time.time()
        while time.time() - started < COLLECT_TIMEOUT:
            state = analyst.get("/data/job.json").json()
            if state["status"] != "running":
                break
            time.sleep(2)
        seconds = round(time.time() - started)
        passed = state["status"] == "done" and f"организациях — {COLLECT_LIMIT}, ошибок — 0" in state["message"] \
            and "рабочим не сделан" in state["message"]
        return f"состояние «{state['status']}» за {seconds} с, сообщение «{state['message']}»", passed
    protocol.run(f"Получение данных с сайта {config.MONITORING_BASE_URL} за {year} год (аналитик, "
                 f"{COLLECT_LIMIT} организации)",
                 f"состояние «done», сообщение «Готово: записей об организациях — {COLLECT_LIMIT}, ошибок — 0. "
                 f"Набор сохранён, но рабочим не сделан…»", collect)

    def dataset_listed():
        listed = "Сбор с сайта мониторинга от" in analyst.get("/data/").text
        return f"набор «Сбор с сайта мониторинга» {'есть в перечне' if listed else 'не найден'}", listed
    protocol.run("Собранный набор данных появился в разделе «Данные»",
                 "набор «Сбор с сайта мониторинга» есть в перечне", dataset_listed)

    def by_link():
        analyst.get("/data/")
        analyst.post("/data/link", {"url": config.DATASET_URL, "name": "Проверка загрузки по ссылке"})
        started = time.time()
        while time.time() - started < LINK_TIMEOUT:
            state = analyst.get("/data/job.json").json()
            if state["status"] != "running":
                break
            time.sleep(2)
        seconds = round(time.time() - started)
        listed = "<td>ссылка</td>" in analyst.get("/data/").text
        passed = state["status"] == "done" and "загружен по ссылке" in state["message"] and listed
        return f"состояние «{state['status']}» за {seconds} с, сообщение «{state['message']}», набор с источником " \
               f"«ссылка» {'есть в перечне' if listed else 'не найден'}", passed
    protocol.run(f"Загрузка набора данных по ссылке {config.DATASET_URL} (аналитик)",
                 "состояние «done», сообщение «Готово: набор данных №N загружен по ссылке, записей об организациях — "
                 "M», набор с источником «ссылка» есть в перечне", by_link)

    def still_working():
        after = working_dataset(analyst)
        code = analyst.get("/").status_code
        return f"рабочий набор {after}, страница «Обзор» — код {code}", after == before and code == 200
    protocol.run("После пробного сбора и загрузки по ссылке рабочий набор данных прежний",
                 f"рабочий набор {before}, страница «Обзор» — код 200", still_working)


def wait_for_application(url: str, seconds: int) -> str:
    """
    Ждёт, пока по адресу начнёт отвечать именно эта система. Возвращает пустую строку
    или описание того, что получено вместо страницы входа.
    """
    deadline = time.time() + seconds
    while True:
        try:
            response = Visitor(url).get("/login")
            # Проверяем не только код ответа: порт может быть занят другой программой
            if response.status_code == 200 and 'name="_csrf"' in response.text:
                return ""
            answer = f"отвечает другая программа (код {response.status_code}, сервер " \
                     f"«{response.headers.get('Server', 'не указан')}»)"
        except requests.RequestException as error:
            answer = f"нет ответа ({type(error).__name__})"
        if time.time() >= deadline:
            return answer
        time.sleep(2)


def main() -> int:
    parser = argparse.ArgumentParser(description="Проверка установки системы по основным сценариям работы")
    parser.add_argument("--url", default="http://127.0.0.1:5000", help="адрес запущенного приложения")
    parser.add_argument("--collect", action="store_true",
                        help="проверить получение данных с сайта мониторинга и по ссылке из репозитория "
                             "(нужен доступ к Интернету)")
    parser.add_argument("--report", default="", help="файл, в который сохранить протокол (Markdown)")
    parser.add_argument("--wait", type=int, default=0, metavar="СЕКУНД",
                        help="сколько секунд ждать запуска приложения (по умолчанию не ждать)")
    args = parser.parse_args()

    answer = wait_for_application(args.url, args.wait)
    if answer:
        print(f"Система по адресу {args.url} не найдена: {answer}.\n"
              f"Запустите её командой «python main.py» и повторите проверку. Если порт занят другой программой\n"
              f"(в macOS порт 5000 использует служба AirPlay), запустите систему на другом порту:\n"
              f"«python main.py --port 8080» и «python check_install.py --url http://127.0.0.1:8080».")
        return 1

    protocol = Protocol()
    check_entrance(protocol, args.url)
    visitors = {role: check_role(protocol, args.url, role) for role in ACCOUNTS}
    check_forecast(protocol, visitors["analyst"], visitors["viewer"])
    check_documents(protocol, visitors["viewer"], visitors["admin"])
    check_protection(protocol, args.url, visitors["viewer"], visitors["admin"])
    if args.collect:
        check_collect(protocol, visitors["analyst"])

    total = len(protocol.rows)
    print(f"\nСценариев: {total}, выполнено: {total - protocol.failed}, расхождений: {protocol.failed}")
    if args.report:
        header = [f"Адрес приложения: {args.url}",
                  f"Python {platform.python_version()}, {platform.system()} {platform.release()}",
                  f"Дата проверки: {time.strftime('%d.%m.%Y %H:%M')}"]
        with open(args.report, "w", encoding="utf-8") as handle:
            handle.write(protocol.markdown(header))
        print(f"Протокол сохранён: {args.report}")
    return 1 if protocol.failed else 0


if __name__ == "__main__":
    sys.exit(main())
