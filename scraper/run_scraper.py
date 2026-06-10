"""
Ponto de entrada do scraper — COLETA COMPLETA.

Percorre todas as ligas e temporadas configuradas em config.py e
coleta todas as partidas encerradas com suas estatisticas.

Uso:
    conda activate oscabet-scraper
    python run_scraper.py
"""
import os
import sys
import time

# Garante que 'config' e o pacote 'src' sejam importaveis
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.client import SofascoreClient   # noqa: E402
from src.collector import collect_all     # noqa: E402
from src.logger import get_logger         # noqa: E402
from src.storage import Storage           # noqa: E402

log = get_logger("run")


def main():
    log.info("=" * 60)
    log.info("OscaBet — coleta COMPLETA de dados do Sofascore")
    log.info("=" * 60)
    inicio = time.time()

    client = SofascoreClient()
    storage = Storage()

    total = collect_all(client, storage)

    minutos = (time.time() - inicio) / 60
    log.info(f"Concluido em {minutos:.1f} minutos. {total} partidas novas.")


if __name__ == "__main__":
    main()
