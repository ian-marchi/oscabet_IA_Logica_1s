"""
Configuracao central do scraper do Sofascore.

Concentra em um unico lugar: caminhos de arquivos, lista de ligas,
temporadas desejadas e parametros de rate limiting. Assim, qualquer
ajuste de escopo da coleta e feito aqui, sem mexer no codigo.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Raiz REAL do projeto: sobe a árvore até achar a pasta que contém 'agent' e 'data'
# (o scraper fica em <raiz>/Banco de dados/scraper, então não basta parent.parent).
def _find_project_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "agent").is_dir() and (p / "data").is_dir():
            return p
    # fallback: dois níveis acima (layout antigo, scraper na raiz)
    return start.parent.parent

PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
load_dotenv(PROJECT_ROOT / ".env")

# ------------------------------------------------------------------
# Caminhos dos arquivos de dados
# ------------------------------------------------------------------
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
MATCHES_CSV = RAW_DATA_DIR / "matches.csv"
MATCH_STATS_CSV = RAW_DATA_DIR / "match_stats.csv"
SCRAPER_LOG_JSON = RAW_DATA_DIR / "scraper_log.json"
SCRAPER_LOG_TXT = RAW_DATA_DIR / "scraper_run.log"

# ------------------------------------------------------------------
# Ligas a coletar — chave interna -> id do torneio no Sofascore
# ------------------------------------------------------------------
LEAGUES = {
    "brasileirao_a": 325,
    "brasileirao_b": 390,
    "copa_brasil": 162,
    "libertadores": 384,
    "premier_league": 17,
    "la_liga": 8,
    "serie_a": 23,
    "bundesliga": 35,
    "ligue_1": 34,
    "champions_league": 7,
    # ── Seleções (para previsões da Copa do Mundo 2026) ──
    "amistosos": 851,      # Int. Friendly Games (forma das seleções)
    "copa_mundo": 16,      # FIFA World Cup 2026
}

# Quantas temporadas recentes coletar por liga.
# A API devolve as temporadas da mais nova para a mais antiga; pegamos
# as N primeiras. Isso funciona tanto para ligas europeias (rotulo
# "24/25") quanto brasileiras (rotulo "2024"), sem depender do formato.
NUM_SEASONS = int(os.getenv("SCRAPER_NUM_SEASONS", "6"))

# ------------------------------------------------------------------
# Rate limiting — atraso aleatorio entre requisicoes (segundos)
# ------------------------------------------------------------------
DELAY_MIN = float(os.getenv("SCRAPER_DELAY_MIN", "2.0"))
DELAY_MAX = float(os.getenv("SCRAPER_DELAY_MAX", "4.0"))

# Coletar o endpoint /incidents como fonte extra de cartoes?
# Desligado por padrao: o endpoint /statistics ja traz os cartoes
# amarelos e vermelhos, entao a requisicao extra seria dispensavel e
# dobraria o tempo total de coleta. Ligue apenas se quiser redundancia.
FETCH_INCIDENTS = False

# Codigo de status do Sofascore para partida encerrada
FINISHED_STATUS_CODE = 100

# Numero maximo de paginas de eventos por temporada (protecao contra loop)
MAX_PAGES_PER_SEASON = 40
