# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Запуск сбора данных из командной строки:

    python -m vuzmon.collect --years 2021 2022 2023 2024 2025 --workers 8

Сборщик обходит сайт мониторинга, складывает HTML-страницы в кэш и
формирует CSV-файлы набора данных в каталоге data/monitoring.
"""

import argparse
import logging
import sys
import time

from vuzmon import config
from vuzmon.collect.crawler import MonitoringCrawler, records_to_frames, save_frames
from vuzmon.collect.http_client import CachedHttpClient


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Сбор данных мониторинга эффективности вузов")
    parser.add_argument("--years", type=int, nargs="+", default=list(config.DEFAULT_YEARS),
                        help="годы мониторинга (по умолчанию 2021–2025)")
    parser.add_argument("--workers", type=int, default=8, help="число потоков скачивания")
    parser.add_argument("--interval", type=float, default=0.25,
                        help="минимальная пауза между запросами, с")
    parser.add_argument("--limit", type=int, default=None,
                        help="ограничить число организаций в каждом году (для проверки)")
    parser.add_argument("--out", default=str(config.MONITORING_DIR), help="каталог для CSV-файлов")
    parser.add_argument("--cache", default=str(config.CACHE_DIR), help="каталог кэша HTML-страниц")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("collect")

    client = CachedHttpClient(args.cache, min_interval=args.interval)
    crawler = MonitoringCrawler(client, workers=args.workers,
                                progress=lambda done, total, message: log.info(message))
    started = time.time()
    result = crawler.crawl(args.years, limit=args.limit)
    frames = records_to_frames(result.records)
    paths = save_frames(frames, args.out)

    log.info("Готово за %.0f с. Записей об организациях: %s", time.time() - started,
             len(frames["institutions"]))
    log.info("Статистика загрузки: %s", client.stats)
    for name, path in paths.items():
        log.info("  %-12s -> %s (%s строк)", name, path, len(frames[name]))
    if result.errors:
        log.warning("Ошибок при сборе: %s", len(result.errors))
        for line in result.errors[:20]:
            log.warning("  %s", line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
