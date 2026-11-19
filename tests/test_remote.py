# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Тесты получения набора данных по ссылке. К Интернету тесты не обращаются:
файлы набора раздаёт временный HTTP-сервер, запущенный на этом же компьютере.
"""

import functools
import http.server
import threading
import zipfile

import pytest

from vuzmon import config
from vuzmon.collect import remote, validation
from vuzmon.collect.crawler import DATASET_FILES, load_frames_from_url, save_frames


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    """Раздаёт файлы каталога и не печатает строку о каждом запросе."""

    def log_message(self, *args) -> None:
        pass


@pytest.fixture()
def file_server(tmp_path, small_frames, monkeypatch):
    """Адрес временного сервера, на котором лежат набор данных (каталог и архив) и испорченные файлы."""
    # Обращения к собственному компьютеру не должны уходить через системный прокси
    monkeypatch.setenv("NO_PROXY", "127.0.0.1")
    monkeypatch.setenv("no_proxy", "127.0.0.1")
    save_frames(small_frames, tmp_path / "set")
    with zipfile.ZipFile(tmp_path / "set.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for filename in DATASET_FILES.values():
            archive.write(tmp_path / "set" / filename, f"monitoring/{filename}")
    (tmp_path / "broken.zip").write_bytes(b"not a zip")
    (tmp_path / "partial").mkdir()
    (tmp_path / "partial" / "institutions.csv").write_bytes(b"")
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(QuietHandler, directory=str(tmp_path)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def test_dataset_is_loaded_by_link(file_server, small_frames):
    # Требование задания: данные для обучения можно получить напрямую по ссылке
    sizes = {name: len(frame) for name, frame in small_frames.items()}
    steps = []
    from_folder = remote.load_frames_from_url(f"{file_server}/set/", progress=lambda *step: steps.append(step))
    assert {name: len(frame) for name, frame in from_folder.items()} == sizes
    assert len(steps) == len(DATASET_FILES) and "institutions.csv" in steps[0][2]
    assert validation.check_frames(from_folder) == []
    from_archive = load_frames_from_url(f"{file_server}/set.zip")          # тот же вызов, что в блокноте
    assert {name: len(frame) for name, frame in from_archive.items()} == sizes
    assert list(from_archive["institutions"].columns) == list(small_frames["institutions"].columns)


def test_link_problems_are_explained(file_server, monkeypatch):
    with pytest.raises(remote.DatasetDownloadError, match="кодом 404"):
        remote.load_frames_from_url(f"{file_server}/missing")
    with pytest.raises(remote.DatasetDownloadError, match="http:// или https://"):
        remote.load_frames_from_url("ftp://example.org/set")
    with pytest.raises(remote.DatasetDownloadError, match="не является ZIP-архивом"):
        remote.load_frames_from_url(f"{file_server}/broken.zip")
    with pytest.raises(ValueError, match="файл institutions.csv пуст"):
        remote.read_csv_table({"institutions.csv": b""}, "institutions.csv")
    monkeypatch.setattr(remote, "RETRY_PAUSE", 0.0)
    with pytest.raises(remote.DatasetDownloadError, match="после 3 попыток"):
        remote.load_frames_from_url("http://127.0.0.1:9/set", timeout=2)   # порт, на котором никто не отвечает
    monkeypatch.setattr(remote, "MAX_DOWNLOAD_BYTES", 2000)
    with pytest.raises(remote.DatasetDownloadError, match="больше допустимого размера"):
        remote.load_frames_from_url(f"{file_server}/set")


def test_broken_connection_is_retried(file_server, small_frames, monkeypatch):
    # Найдено при проверке на опубликованном наборе: единичный обрыв соединения сразу приводил к отказу
    monkeypatch.setattr(remote, "RETRY_PAUSE", 0.0)
    original, calls = remote.requests.Session.get, []

    def flaky(self, url, **kwargs):
        calls.append(url)
        if len(calls) in (1, 4):                      # первый запрос и ещё один в середине обрываются
            raise remote.requests.exceptions.SSLError("обрыв соединения")
        return original(self, url, **kwargs)
    monkeypatch.setattr(remote.requests.Session, "get", flaky)
    frames = remote.load_frames_from_url(f"{file_server}/set")
    assert len(frames["institutions"]) == len(small_frames["institutions"])
    assert len(calls) == len(DATASET_FILES) + 2       # два оборванных запроса повторены


def test_github_page_link_is_converted_to_raw_files():
    page = "https://github.com/sinkarev39/vuz-monitoring-analysis/tree/main/data/monitoring/"
    assert remote.normalize_url(page) == config.DATASET_URL
    assert remote.normalize_url(config.DATASET_URL + "/") == config.DATASET_URL
    assert remote.normalize_url("  https://example.org/data.zip ") == "https://example.org/data.zip"
