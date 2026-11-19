# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль remote — получение набора данных напрямую по ссылке.

Набор данных системы опубликован в её репозитории (адрес — config.DATASET_URL),
поэтому его можно не копировать вручную, а скачать: из блокнота Jupyter, из
раздела «Данные» веб-приложения или из собственной программы. Ссылка может
указывать на каталог с пятью CSV-файлами набора либо на ZIP-архив с ними.

Скачивание идёт с ограничениями: только протоколы http и https, таймаут на
каждый запрос, предельный суммарный размер. Обрыв соединения не считается
отказом сразу: запрос повторяется несколько раз. Все причины отказа описаны
по-русски в исключении DatasetDownloadError.
"""

import io
import re
import time
import zipfile

import requests

from vuzmon.collect.crawler import DATASET_FILES
from vuzmon.collect.http_client import DEFAULT_USER_AGENT
from vuzmon.collect.validation import read_csv_table

MAX_DOWNLOAD_BYTES = 64 * 1024 * 1024       # предельный суммарный размер скачиваемых файлов
CHUNK_BYTES = 64 * 1024
ATTEMPTS = 3                                # число попыток на один файл при обрыве соединения
RETRY_PAUSE = 2.0                           # пауза перед повторной попыткой, с (умножается на номер попытки)


class DatasetDownloadError(ValueError):
    """Набор данных не удалось получить по ссылке (текст исключения — причина на русском языке)."""


def normalize_url(url: str) -> str:
    """
    Приводит ссылку к адресу, по которому отдаются сами файлы. Адрес каталога на
    странице GitHub (https://github.com/<автор>/<репозиторий>/tree/<ветка>/<путь>)
    заменяется адресом «сырых» файлов на raw.githubusercontent.com.
    """
    url = url.strip()
    found = re.fullmatch(r"https://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/(.+?)/?", url)
    if found:
        return "https://raw.githubusercontent.com/{}/{}/{}/{}".format(*found.groups())
    return url.rstrip("/")


def download_dataset(url: str, timeout: float = 60.0, progress=None) -> dict:
    """
    Скачивает файлы набора данных и возвращает словарь {имя файла: содержимое}.

    url      — адрес каталога с CSV-файлами набора или адрес ZIP-архива;
    timeout  — предельное время ожидания ответа на один запрос, с;
    progress — необязательная функция progress(done, total, message) для показа хода работ.
    """
    url = normalize_url(url)
    if not url.lower().startswith(("http://", "https://")):
        raise DatasetDownloadError("ссылка должна начинаться с http:// или https://")
    progress = progress or (lambda done, total, message: None)
    session = requests.Session()
    session.headers.update({"User-Agent": DEFAULT_USER_AGENT})
    received = {"bytes": 0}

    def fetch(address: str) -> bytes:
        """Один файл; при обрыве соединения запрос повторяется, при отказе сервера — нет."""
        for attempt in range(1, ATTEMPTS + 1):
            before = received["bytes"]
            try:
                with session.get(address, timeout=timeout, stream=True) as response:
                    if response.status_code != 200:
                        raise DatasetDownloadError(
                            f"сервер ответил кодом {response.status_code} на запрос {address}")
                    chunks = []
                    for chunk in response.iter_content(CHUNK_BYTES):
                        received["bytes"] += len(chunk)
                        if received["bytes"] > MAX_DOWNLOAD_BYTES:
                            raise DatasetDownloadError(
                                f"набор данных больше допустимого размера ({MAX_DOWNLOAD_BYTES // 2 ** 20} МБ)")
                        chunks.append(chunk)
                    return b"".join(chunks)
            except requests.RequestException as error:
                received["bytes"] = before          # недокачанная часть в общий размер не входит
                if attempt == ATTEMPTS:
                    raise DatasetDownloadError(f"не удалось получить {address} после {ATTEMPTS} попыток "
                                               f"({type(error).__name__})") from error
                time.sleep(RETRY_PAUSE * attempt)

    if url.lower().split("?")[0].endswith(".zip"):
        progress(0, 1, "скачивание архива")
        try:
            with zipfile.ZipFile(io.BytesIO(fetch(url))) as archive:
                contents = {member.split("/")[-1].lower(): archive.read(member) for member in archive.namelist()
                            if member.split("/")[-1].lower() in DATASET_FILES.values()}
        except zipfile.BadZipFile:
            raise DatasetDownloadError("файл по ссылке не является ZIP-архивом или повреждён") from None
        return contents

    contents = {}
    files = list(DATASET_FILES.values())
    for number, filename in enumerate(files):
        progress(number, len(files), f"файл {filename} ({number + 1} из {len(files)})")
        contents[filename] = fetch(f"{url}/{filename}")
    return contents


def load_frames_from_url(url: str, timeout: float = 60.0, progress=None) -> dict:
    """
    Загружает набор данных по ссылке и возвращает таблицы pandas — в том же
    виде, что и load_frames для каталога на диске. При ошибке выбрасывает
    ValueError (DatasetDownloadError — если файлы не удалось скачать).
    """
    contents = download_dataset(url, timeout=timeout, progress=progress)
    return {table: read_csv_table(contents, filename) for table, filename in DATASET_FILES.items()}
