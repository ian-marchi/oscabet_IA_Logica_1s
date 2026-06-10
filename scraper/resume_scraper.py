"""
Ponto de entrada do scraper — COLETA INCREMENTAL (retoma de onde parou).

Le o arquivo de estado scraper_log.json para descobrir a data da
partida mais recente ja coletada de cada liga, e busca somente as
partidas posteriores a essa data — evitando reprocessar o banco todo.

Uso:
    conda activate oscabet-scraper
    python resume_scraper.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import LEAGUES                # noqa: E402
from src.client import SofascoreClient    # noqa: E402
from src.collector import collect_league  # noqa: E402
from src.logger import get_logger         # noqa: E402
from src.storage import Storage           # noqa: E402

log = get_logger("resume")


def last_date_for_league(state: dict, league_name: str):
    """Acha a data da partida mais recente ja coletada de uma liga."""
    temporadas = state.get("leagues", {}).get(league_name, {})
    datas = [info.get("last_match_date") for info in temporadas.values()
             if info.get("last_match_date")]
    return max(datas) if datas else None


def main():
    log.info("=" * 60)
    log.info("OscaBet — coleta INCREMENTAL de dados do Sofascore")
    log.info("=" * 60)
    inicio = time.time()

    client = SofascoreClient()
    storage = Storage()
    state = storage.read_log()

    total = 0
    for league_name, league_id in LEAGUES.items():
        since = last_date_for_league(state, league_name)
        log.info(f"Liga {league_name}: partidas apos {since or '(inicio)'}")
        try:
            total += collect_league(client, storage, league_name,
                                    league_id, since_date=since)
        except Exception as e:
            log.error(f"Erro na liga {league_name}: {e}")

    storage.finish_run(client.request_count)
    minutos = (time.time() - inicio) / 60
    log.info(f"Retomada concluida em {minutos:.1f} min. {total} partidas novas.")


if __name__ == "__main__":
    main()
