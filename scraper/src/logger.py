"""
Logger de progresso e erros do scraper.

Escreve as mensagens tanto no console (para acompanhar a coleta em
tempo real) quanto em um arquivo de log (para auditar depois). Usa o
modulo padrao `logging` do Python.
"""
import logging
import sys

from config import SCRAPER_LOG_TXT

_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Cache de loggers ja criados, para nao duplicar handlers
_loggers: dict[str, logging.Logger] = {}


def get_logger(name: str) -> logging.Logger:
    """
    Devolve um logger configurado. Se o logger com esse nome ja
    existir, reaproveita (evita imprimir cada mensagem varias vezes).
    """
    if name in _loggers:
        return _loggers[name]

    logger = logging.getLogger(f"scraper.{name}")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATE_FORMAT)

    # Handler de console
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Handler de arquivo
    try:
        SCRAPER_LOG_TXT.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(SCRAPER_LOG_TXT, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    except OSError:
        # Se nao conseguir criar o arquivo, segue so com console
        pass

    _loggers[name] = logger
    return logger
