# -*- coding: utf-8 -*-
"""
Автор: Шинкарев Сергей Дмитриевич, группа о.ИЗДт 23.2/Б3-22
Тема ВКР: «Разработка системы интеллектуального анализа данных мониторинга вузов»

Точка входа: запуск веб-приложения системы.

    python main.py                  — запуск на http://127.0.0.1:5000
    python main.py --port 8080      — другой порт
    python main.py --host 0.0.0.0   — доступ с других компьютеров сети

После запуска откройте адрес в браузере и войдите под одной из учётных
записей, перечисленных в файле README.md.
"""

import argparse
import logging

from webapp import create_app


def main() -> None:
    parser = argparse.ArgumentParser(description="Система интеллектуального анализа данных мониторинга вузов")
    parser.add_argument("--host", default="127.0.0.1", help="адрес, на котором принимаются подключения")
    parser.add_argument("--port", type=int, default=5000, help="порт веб-сервера")
    parser.add_argument("--debug", action="store_true", help="режим отладки (только для разработки)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    app = create_app()
    print(f"Система запущена: http://{args.host}:{args.port}  (остановка — Ctrl+C)")
    # Многопоточный режим нужен, чтобы страница отвечала во время фонового обучения
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True, use_reloader=False)


if __name__ == "__main__":
    main()
