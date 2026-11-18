# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Модуль http_client — «вежливый» HTTP-клиент для сбора открытых данных.

Особенности:
  * каждая скачанная страница сохраняется в дисковый кэш (gzip), поэтому
    повторный запуск сборщика не создаёт нагрузку на сайт-источник;
  * между запросами выдерживается пауза (ограничение частоты обращений);
  * при таймаутах и ошибках сервера запрос повторяется с нарастающей задержкой.
"""

import gzip
import logging
import threading
import time
from pathlib import Path

import requests

logger = logging.getLogger(__name__)

# Обезличенный идентификатор клиента: персональные данные в заголовки не попадают
DEFAULT_USER_AGENT = "Mozilla/5.0 (compatible; VuzMonitoringResearch/1.0; educational project)"


class DownloadError(RuntimeError):
    """Страницу не удалось получить после всех повторных попыток."""


class CachedHttpClient:
    """HTTP-клиент с дисковым кэшем, ограничением частоты и повторами."""

    def __init__(self, cache_dir, min_interval: float = 0.25, timeout: float = 60.0,
                 max_retries: int = 6, user_agent: str = DEFAULT_USER_AGENT):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval      # минимальная пауза между стартами запросов, с
        self.timeout = timeout                # таймаут одного запроса, с
        self.max_retries = max_retries        # число попыток на одну страницу
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Language": "ru"})
        # Блокировка нужна, потому что клиент используется из нескольких потоков
        self._lock = threading.Lock()
        self._last_request_at = 0.0
        # Простая статистика работы для итогового протокола сбора
        self.stats = {"downloaded": 0, "from_cache": 0, "retries": 0, "failed": 0}

    # ------------------------------------------------------------------ кэш
    def _cache_path(self, cache_key: str) -> Path:
        """Путь к файлу кэша; ключ вида '2025/inst_1544' превращается в подкаталоги."""
        return self.cache_dir / f"{cache_key}.html.gz"

    def read_cache(self, cache_key: str):
        """Возвращает текст страницы из кэша или None, если её там нет."""
        path = self._cache_path(cache_key)
        if not path.exists():
            return None
        try:
            with gzip.open(path, "rt", encoding="utf-8") as handle:
                return handle.read()
        except (OSError, EOFError):
            # Повреждённый файл (например, прерванная запись) просто удаляем
            path.unlink(missing_ok=True)
            return None

    def _write_cache(self, cache_key: str, text: str) -> None:
        path = self._cache_path(cache_key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Пишем во временный файл и переименовываем: запись получается атомарной
        tmp_path = path.with_suffix(".tmp")
        with gzip.open(tmp_path, "wt", encoding="utf-8") as handle:
            handle.write(text)
        tmp_path.replace(path)

    # -------------------------------------------------------------- запросы
    def _wait_turn(self) -> None:
        """Выдерживает паузу между запросами (общую для всех потоков)."""
        with self._lock:
            pause = self._last_request_at + self.min_interval - time.monotonic()
            if pause > 0:
                time.sleep(pause)
            self._last_request_at = time.monotonic()

    def get(self, url: str, cache_key: str, must_contain: str = "", force: bool = False) -> str:
        """
        Возвращает HTML-страницу по адресу url.

        cache_key    — ключ дискового кэша;
        must_contain — подстрока, которая обязана быть в корректном ответе
                       (защита от «пустых» ответов и страниц-заглушек);
        force        — игнорировать кэш и скачать заново.
        """
        if not force:
            cached = self.read_cache(cache_key)
            if cached is not None:
                self.stats["from_cache"] += 1
                return cached

        last_error = None
        for attempt in range(1, self.max_retries + 1):
            self._wait_turn()
            try:
                response = self.session.get(url, timeout=self.timeout)
                if response.status_code >= 500:
                    raise requests.HTTPError(f"HTTP {response.status_code}")
                response.raise_for_status()
                response.encoding = "utf-8"
                text = response.text
                if must_contain and must_contain not in text:
                    raise ValueError("в ответе нет ожидаемого содержимого")
                self._write_cache(cache_key, text)
                self.stats["downloaded"] += 1
                return text
            except (requests.RequestException, ValueError) as error:
                last_error = error
                self.stats["retries"] += 1
                # Нарастающая задержка: 2, 4, 8 ... секунд, но не более 30
                delay = min(30.0, 2.0 ** attempt)
                logger.debug("Попытка %s для %s не удалась (%s), пауза %.0f с",
                             attempt, url, error, delay)
                time.sleep(delay)

        self.stats["failed"] += 1
        raise DownloadError(f"Не удалось получить {url}: {last_error}")
