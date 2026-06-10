"""
Coleta e parsing das estatisticas de uma partida especifica.

A API do Sofascore devolve as estatisticas agrupadas e com nomes em
ingles. Este modulo extrai os numeros e os traduz para as colunas
padronizadas de match_stats.csv. Os cartoes vem do endpoint de
incidentes, que e a fonte mais confiavel para contagem de cartoes.
"""
from config import FETCH_INCIDENTS
from .endpoints import match_incidents_url, match_stats_url
from .logger import get_logger

log = get_logger("match_scraper")

# Mapa: nome da estatistica no Sofascore -> sufixo da coluna no CSV.
# Cada sufixo gera duas colunas: home_<sufixo> e away_<sufixo>.
STAT_NAME_MAP = {
    "Ball possession": "possession",
    "Total shots": "shots",
    "Shots on target": "shots_on_target",
    "Shots off target": "shots_off_target",
    "Blocked shots": "shots_blocked",
    "Big chances": "big_chances",
    "Big chances missed": "big_chances_missed",
    "Corner kicks": "corners",
    "Fouls": "fouls",
    "Offsides": "offsides",
    "Passes": "passes",
    "Accurate passes": "passes_accurate",
    "Tackles": "tackles",
    "Tackles won": "tackles",
    "Interceptions": "interceptions",
    "Clearances": "clearances",
    "Total clearances": "clearances",
    "Goalkeeper saves": "saves",
    "Total saves": "saves",
    "Expected goals": "xg",
    "Dribbles": "dribbles_successful",
    "Successful dribbles": "dribbles_successful",
    "Yellow cards": "yellow_cards",
    "Red cards": "red_cards",
}


def _to_number(value):
    """
    Converte um valor de estatistica para numero.
    Trata '55%', '12', '450/520 (87%)', '8 (3)'. Devolve None se falhar.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return value
    texto = str(value).strip()
    if texto == "":
        return None
    if "/" in texto:                 # '450/520' -> usa o primeiro numero
        texto = texto.split("/")[0]
    if "(" in texto:                 # '8 (3)' -> usa o primeiro numero
        texto = texto.split("(")[0]
    texto = texto.replace("%", "").strip()
    try:
        return float(texto) if "." in texto else int(texto)
    except ValueError:
        return None


def _split_won_total(texto):
    """De um texto como '10/18 (56%)' devolve a tupla (ganhos, total)."""
    if texto is None:
        return (None, None)
    base = str(texto).split("(")[0].strip()
    if "/" in base:
        partes = base.split("/")
        return (_to_number(partes[0]), _to_number(partes[1]))
    return (_to_number(base), None)


def _soma(a, b):
    """Soma dois valores tratando None como ausencia (devolve None se ambos faltam)."""
    if a is None and b is None:
        return None
    return (a or 0) + (b or 0)


def parse_statistics(stats_json: dict) -> dict:
    """
    Recebe o JSON de /statistics e devolve um dict com as colunas
    home_* e away_* preenchidas. Usa apenas o periodo 'ALL' (jogo inteiro).
    """
    resultado = {}
    if not stats_json or "statistics" not in stats_json:
        return resultado

    periodo_all = None
    for periodo in stats_json["statistics"]:
        if periodo.get("period") == "ALL":
            periodo_all = periodo
            break
    if periodo_all is None:
        return resultado

    for grupo in periodo_all.get("groups", []):
        for item in grupo.get("statisticsItems", []):
            nome = item.get("name", "")

            # Duelos aereos: o texto traz 'ganhos/total'
            if nome in ("Aerial duels won", "Aerial duels", "Aerials won"):
                hw, ht = _split_won_total(item.get("home"))
                aw, at = _split_won_total(item.get("away"))
                resultado["home_aerial_duels_won"] = hw
                resultado["away_aerial_duels_won"] = aw
                resultado["home_aerial_duels_total"] = ht
                resultado["away_aerial_duels_total"] = at
                continue

            sufixo = STAT_NAME_MAP.get(nome)
            if not sufixo:
                continue

            # Prefere os campos numericos; cai para o texto se faltarem
            home = item.get("homeValue", item.get("home"))
            away = item.get("awayValue", item.get("away"))
            resultado[f"home_{sufixo}"] = _to_number(home)
            resultado[f"away_{sufixo}"] = _to_number(away)

    return resultado


def parse_incidents(incidents_json: dict) -> dict:
    """
    Recebe o JSON de /incidents e conta os cartoes de cada lado.
    Devolve um dict com cartoes amarelos e vermelhos por mando.
    """
    contagem = {
        "home_yellow_cards": 0, "away_yellow_cards": 0,
        "home_red_cards": 0, "away_red_cards": 0,
    }
    if not incidents_json or "incidents" not in incidents_json:
        return {}

    for inc in incidents_json["incidents"]:
        if inc.get("incidentType") != "card":
            continue
        lado = "home" if inc.get("isHome") else "away"
        classe = (inc.get("incidentClass") or "").lower()
        if classe == "yellow":
            contagem[f"{lado}_yellow_cards"] += 1
        elif classe in ("red", "yellowred"):
            contagem[f"{lado}_red_cards"] += 1
            if classe == "yellowred":   # 2o amarelo tambem conta como amarelo
                contagem[f"{lado}_yellow_cards"] += 1
    return contagem


def scrape_match_stats(client, match_id: int):
    """
    Coleta as estatisticas completas de uma partida.

    A fonte principal e o endpoint /statistics, que ja inclui os
    cartoes amarelos e vermelhos. Como esse endpoint so lista o que
    de fato ocorreu, a ausencia de uma estatistica de cartao significa
    que houve ZERO daquele cartao — por isso o valor ausente vira 0.

    Opcionalmente (config FETCH_INCIDENTS), tambem consulta /incidents
    como fonte alternativa de contagem de cartoes.

    Devolve um dict pronto para virar uma linha de match_stats.csv,
    ou None quando a partida nao tem estatisticas disponiveis.
    """
    stats = parse_statistics(client.get_json(match_stats_url(match_id)))
    if not stats:
        log.warning(f"Partida {match_id}: sem estatisticas disponiveis.")
        return None

    row = {"match_id": int(match_id)}
    row.update(stats)

    # /statistics so lista o que ocorreu: cartao ausente = 0 cartoes
    for campo in ("home_yellow_cards", "away_yellow_cards",
                  "home_red_cards", "away_red_cards"):
        if row.get(campo) is None:
            row[campo] = 0

    # Fonte alternativa de cartoes (desligada por padrao via config)
    if FETCH_INCIDENTS:
        cartoes = parse_incidents(client.get_json(match_incidents_url(match_id)))
        if cartoes:
            row.update(cartoes)

    # Totais derivados
    row["total_corners"] = _soma(row.get("home_corners"), row.get("away_corners"))
    row["total_yellow_cards"] = _soma(
        row.get("home_yellow_cards"), row.get("away_yellow_cards"))
    return row
