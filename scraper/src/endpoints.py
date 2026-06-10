"""
URLs da API publica do Sofascore, centralizadas em funcoes.

Manter as URLs isoladas aqui evita repeticao de strings pelo codigo
e facilita corrigir um endpoint caso o Sofascore mude a API.
"""

BASE = "https://api.sofascore.com/api/v1"


def seasons_url(league_id: int) -> str:
    """Lista de temporadas disponiveis de uma competicao."""
    return f"{BASE}/unique-tournament/{league_id}/seasons"


def events_last_url(league_id: int, season_id: int, page: int) -> str:
    """Pagina de partidas JA REALIZADAS de uma temporada (page comeca em 0)."""
    return (f"{BASE}/unique-tournament/{league_id}/season/{season_id}"
            f"/events/last/{page}")


def events_next_url(league_id: int, season_id: int, page: int) -> str:
    """Pagina de partidas FUTURAS de uma temporada (usada pelo agente)."""
    return (f"{BASE}/unique-tournament/{league_id}/season/{season_id}"
            f"/events/next/{page}")


def match_stats_url(match_id: int) -> str:
    """Estatisticas completas de uma partida."""
    return f"{BASE}/event/{match_id}/statistics"


def match_incidents_url(match_id: int) -> str:
    """Incidentes de uma partida (gols, cartoes, substituicoes)."""
    return f"{BASE}/event/{match_id}/incidents"


def match_detail_url(match_id: int) -> str:
    """Detalhe geral de uma partida."""
    return f"{BASE}/event/{match_id}"
