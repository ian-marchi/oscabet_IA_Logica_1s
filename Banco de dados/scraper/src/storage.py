"""
Armazenamento incremental dos dados coletados.

Mantem dois CSVs (matches.csv e match_stats.csv) e um arquivo de estado
(scraper_log.json). A escrita e incremental: cada partida coletada e
acrescentada na hora, de modo que uma interrupcao nao perde o progresso.
"""
import json
from datetime import datetime

import pandas as pd

from config import (MATCH_STATS_CSV, MATCHES_CSV, RAW_DATA_DIR,
                    SCRAPER_LOG_JSON)
from .logger import get_logger

log = get_logger("storage")

# Colunas fixas de matches.csv (uma linha por partida)
MATCHES_COLUMNS = [
    "match_id", "league", "season", "date",
    "home_team", "away_team", "home_team_id", "away_team_id",
    "home_score", "away_score", "result", "stats_collected",
]

# Colunas fixas de match_stats.csv (estatisticas detalhadas por partida)
MATCH_STATS_COLUMNS = [
    "match_id",
    "home_possession", "away_possession",
    "home_shots", "away_shots",
    "home_shots_on_target", "away_shots_on_target",
    "home_shots_off_target", "away_shots_off_target",
    "home_shots_blocked", "away_shots_blocked",
    "home_big_chances", "away_big_chances",
    "home_big_chances_missed", "away_big_chances_missed",
    "home_corners", "away_corners", "total_corners",
    "home_yellow_cards", "away_yellow_cards", "total_yellow_cards",
    "home_red_cards", "away_red_cards",
    "home_fouls", "away_fouls",
    "home_offsides", "away_offsides",
    "home_passes", "away_passes",
    "home_passes_accurate", "away_passes_accurate",
    "home_tackles", "away_tackles",
    "home_interceptions", "away_interceptions",
    "home_clearances", "away_clearances",
    "home_aerial_duels_won", "away_aerial_duels_won",
    "home_aerial_duels_total", "away_aerial_duels_total",
    "home_dribbles_successful", "away_dribbles_successful",
    "home_saves", "away_saves",
    "home_xg", "away_xg",
]


class Storage:
    """Gerencia a leitura e a escrita incremental dos arquivos de dados."""

    def __init__(self):
        RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
        # Carrega na memoria os ids ja salvos, para nao coletar duas vezes
        self.known_match_ids = self._load_ids(MATCHES_CSV)
        self.stats_match_ids = self._load_ids(MATCH_STATS_CSV)
        log.info(
            f"Storage iniciado: {len(self.known_match_ids)} partidas e "
            f"{len(self.stats_match_ids)} estatisticas ja no banco."
        )

    @staticmethod
    def _load_ids(csv_path) -> set:
        """Le os match_id de um CSV existente. Devolve um set (vazio se nao existir)."""
        if not csv_path.exists():
            return set()
        try:
            df = pd.read_csv(csv_path, usecols=["match_id"])
            return set(df["match_id"].astype(int).tolist())
        except Exception as e:
            log.warning(f"Nao consegui ler ids de {csv_path}: {e}")
            return set()

    def match_exists(self, match_id: int) -> bool:
        """Diz se a partida ja foi coletada (usado para coleta incremental)."""
        return int(match_id) in self.known_match_ids

    def save_match(self, match_row: dict) -> bool:
        """Acrescenta uma partida em matches.csv. Devolve False se ja existia."""
        mid = int(match_row["match_id"])
        if mid in self.known_match_ids:
            return False
        self._append_row(MATCHES_CSV, match_row, MATCHES_COLUMNS)
        self.known_match_ids.add(mid)
        return True

    def save_match_stats(self, stats_row: dict) -> bool:
        """Acrescenta estatisticas em match_stats.csv. Devolve False se ja existiam."""
        mid = int(stats_row["match_id"])
        if mid in self.stats_match_ids:
            return False
        self._append_row(MATCH_STATS_CSV, stats_row, MATCH_STATS_COLUMNS)
        self.stats_match_ids.add(mid)
        return True

    @staticmethod
    def _append_row(csv_path, row: dict, columns: list) -> None:
        """Acrescenta uma linha ao CSV, criando o cabecalho se o arquivo for novo."""
        ordered = {col: row.get(col) for col in columns}
        df = pd.DataFrame([ordered])
        write_header = not csv_path.exists()
        df.to_csv(csv_path, mode="a", header=write_header, index=False,
                  encoding="utf-8")

    # ------------------------------------------------------------------
    # Estado da coleta (scraper_log.json) — usado para retomar de onde parou
    # ------------------------------------------------------------------
    def read_log(self) -> dict:
        """Le o arquivo de estado da coleta. Devolve a estrutura padrao se nao existir."""
        if not SCRAPER_LOG_JSON.exists():
            return {"last_run": None, "leagues": {}, "total_requests": 0}
        try:
            with open(SCRAPER_LOG_JSON, "r", encoding="utf-8") as f:
                return json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            log.warning(f"scraper_log.json invalido ({e}). Recomecando o estado.")
            return {"last_run": None, "leagues": {}, "total_requests": 0}

    def record_season(self, league: str, season: str, season_id: int,
                       n_matches: int, last_date: str) -> None:
        """Registra no log que uma temporada de uma liga foi coletada."""
        state = self.read_log()
        state.setdefault("leagues", {}).setdefault(league, {})
        state["leagues"][league][season] = {
            "season_id": season_id,
            "matches_collected": n_matches,
            "last_match_date": last_date,
            "updated_at": datetime.now().isoformat(timespec="seconds"),
        }
        self._write_log(state)

    def finish_run(self, total_requests: int) -> None:
        """Marca o fim de uma execucao do scraper no arquivo de estado."""
        state = self.read_log()
        state["last_run"] = datetime.now().isoformat(timespec="seconds")
        state["total_requests"] = total_requests
        self._write_log(state)

    @staticmethod
    def _write_log(state: dict) -> None:
        """Grava o arquivo de estado da coleta em disco."""
        with open(SCRAPER_LOG_JSON, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
