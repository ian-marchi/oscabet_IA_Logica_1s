"""
Orquestrador da coleta de dados.

Percorre cada liga e cada temporada configuradas, pagina a lista de
partidas ja realizadas, e para cada partida nova coleta as estatisticas
e grava nos CSVs. Suporta coleta incremental atraves do parametro
`since_date` (usado pela retomada e pelo updater).
"""
from datetime import datetime, timezone

from config import (FINISHED_STATUS_CODE, LEAGUES, MAX_PAGES_PER_SEASON,
                     NUM_SEASONS)
from .endpoints import events_last_url, seasons_url
from .logger import get_logger
from .match_scraper import scrape_match_stats

log = get_logger("collector")


def resolve_seasons(client, league_id: int) -> list:
    """
    Descobre as temporadas recentes de uma liga.

    A API devolve as temporadas da mais nova para a mais antiga, entao
    pegamos as NUM_SEASONS primeiras. Devolve uma lista de tuplas
    (rotulo, season_id).
    """
    data = client.get_json(seasons_url(league_id))
    if not data or "seasons" not in data:
        return []
    recentes = data["seasons"][:NUM_SEASONS]
    return [(s.get("year"), s.get("id")) for s in recentes]


def parse_event(event: dict, league_name: str, season_label: str):
    """
    Converte um evento da API numa linha de matches.csv.
    Devolve None se a partida nao estiver encerrada ou nao tiver placar.
    """
    status = event.get("status", {})
    if status.get("code") != FINISHED_STATUS_CODE:
        return None

    home_score = event.get("homeScore", {}).get("current")
    away_score = event.get("awayScore", {}).get("current")
    if home_score is None or away_score is None:
        return None

    # Resultado do ponto de vista do mandante: H (vitoria), D (empate), A (derrota)
    if home_score > away_score:
        result = "H"
    elif home_score < away_score:
        result = "A"
    else:
        result = "D"

    ts = event.get("startTimestamp")
    date = None
    if ts:
        date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")

    home = event.get("homeTeam", {})
    away = event.get("awayTeam", {})
    return {
        "match_id": event.get("id"),
        "league": league_name,
        "season": season_label,
        "date": date,
        "home_team": home.get("name"),
        "away_team": away.get("name"),
        "home_team_id": home.get("id"),
        "away_team_id": away.get("id"),
        "home_score": home_score,
        "away_score": away_score,
        "result": result,
        "stats_collected": False,
    }


def collect_season(client, storage, league_name, league_id,
                   season_label, season_id, since_date=None) -> int:
    """
    Coleta uma temporada inteira de uma liga, pagina por pagina.
    Quando `since_date` e informado, para ao alcancar partidas antigas.
    Devolve o numero de partidas novas coletadas.
    """
    novas = 0
    last_date = None
    page = 0

    while page < MAX_PAGES_PER_SEASON:
        data = client.get_json(events_last_url(league_id, season_id, page))
        if not data:
            break
        eventos = data.get("events", [])
        if not eventos:
            break

        parar = False
        for ev in eventos:
            row = parse_event(ev, league_name, season_label)
            if row is None:
                continue

            # Coleta incremental: ignora partidas anteriores a since_date
            if since_date and row["date"] and row["date"] < since_date:
                parar = True
                continue
            if storage.match_exists(row["match_id"]):
                continue

            # Coleta as estatisticas detalhadas da partida
            try:
                stats = scrape_match_stats(client, row["match_id"])
            except Exception as e:
                log.error(f"Erro inesperado ao coletar stats da partida {row['match_id']}: {e}. Pulando.")
                stats = None

            if stats:
                storage.save_match_stats(stats)
                row["stats_collected"] = True

            storage.save_match(row)
            novas += 1
            if last_date is None or (row["date"] and row["date"] > last_date):
                last_date = row["date"]

        if parar or not data.get("hasNextPage", False):
            break
        page += 1

    storage.record_season(league_name, season_label, season_id,
                          novas, last_date or "")
    return novas


def collect_league(client, storage, league_name, league_id,
                   since_date=None) -> int:
    """Coleta todas as temporadas configuradas de uma liga."""
    log.info(f"=== Liga: {league_name} (torneio {league_id}) ===")
    seasons = resolve_seasons(client, league_id)
    if not seasons:
        log.warning(f"Liga {league_name}: nenhuma temporada encontrada.")
        return 0

    total = 0
    for season_label, season_id in seasons:
        log.info(f"  Temporada {season_label} (season_id {season_id})...")
        n = collect_season(client, storage, league_name, league_id,
                           season_label, season_id, since_date)
        log.info(f"  -> {n} partidas novas em {league_name} {season_label}")
        total += n
    return total


def collect_all(client, storage, since_date=None) -> int:
    """Coleta todas as ligas configuradas. Devolve o total de partidas novas."""
    grand_total = 0
    for league_name, league_id in LEAGUES.items():
        try:
            grand_total += collect_league(client, storage, league_name,
                                          league_id, since_date)
        except Exception as e:
            log.error(f"Erro ao coletar a liga {league_name}: {e}")

    storage.finish_run(client.request_count)
    log.info(f"Coleta finalizada: {grand_total} partidas novas no total.")
    log.info(f"Total de requisicoes HTTP realizadas: {client.request_count}")
    return grand_total
