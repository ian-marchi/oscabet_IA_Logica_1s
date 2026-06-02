"""
tools.py — funções que a LLM pode chamar (Tool Use pattern, §4 do plano).

Cada função retorna um dict serializável em JSON.
TOOL_DEFINITIONS: schemas no formato da Claude API.
execute_tool(): dispatcher usado pelo orchestrator.
"""
import pandas as pd
import numpy as np
import data_loader
import predictor as pred_module

# ── Helpers internos ──────────────────────────────────────────────────────────
def _result_for_team(row, team_name: str) -> str:
    """Retorna 'V', 'E' ou 'D' do ponto de vista do time."""
    if row["result"] == "D":
        return "E"
    if row["home_team"] == team_name:
        return "V" if row["result"] == "H" else "D"
    return "V" if row["result"] == "A" else "D"


# ══════════════════════════════════════════════════════════════════════════════
# Tool 1 — get_team_stats
# ══════════════════════════════════════════════════════════════════════════════
def get_team_stats(team_name: str, last_n: int = 10, league: str = None) -> dict:
    """
    Stats agregadas recentes: aproveitamento, gols, cartões, escanteios,
    posse média, forma dos últimos N jogos.
    """
    df   = data_loader.full()
    mask = (df["home_team"] == team_name) | (df["away_team"] == team_name)
    if league:
        mask &= (df["league"] == league)

    hist = df[mask].sort_values("date").tail(last_n)
    if len(hist) == 0:
        return {"error": f"Time '{team_name}' não encontrado."}

    resultados = [_result_for_team(r, team_name) for _, r in hist.iterrows()]

    goals_scored   = [r["home_score"] if r["home_team"] == team_name else r["away_score"] for _, r in hist.iterrows()]
    goals_conceded = [r["away_score"] if r["home_team"] == team_name else r["home_score"] for _, r in hist.iterrows()]
    yellows = [r.get("home_yellow_cards", np.nan) if r["home_team"] == team_name else r.get("away_yellow_cards", np.nan) for _, r in hist.iterrows()]
    corners = [r.get("home_corners", np.nan) if r["home_team"] == team_name else r.get("away_corners", np.nan) for _, r in hist.iterrows()]
    posse   = [r.get("home_possession", np.nan) if r["home_team"] == team_name else r.get("away_possession", np.nan) for _, r in hist.iterrows()]

    total_pts = sum(3 if r == "V" else 1 if r == "E" else 0 for r in resultados)
    form5     = resultados[-5:]
    pts_form5 = sum(3 if r == "V" else 1 if r == "E" else 0 for r in form5)

    return {
        "time":                team_name,
        "liga":                league or "todas",
        "jogos_analisados":    len(hist),
        "aproveitamento_pct":  round(total_pts / (len(resultados) * 3) * 100, 1),
        "vitorias":            resultados.count("V"),
        "empates":             resultados.count("E"),
        "derrotas":            resultados.count("D"),
        "gols_marcados_avg":   round(float(np.mean(goals_scored)), 2),
        "gols_sofridos_avg":   round(float(np.mean(goals_conceded)), 2),
        "amarelos_avg":        round(float(np.nanmean(yellows)), 2),
        "escanteios_avg":      round(float(np.nanmean(corners)), 2),
        "posse_media_pct":     round(float(np.nanmean(posse)), 1),
        "forma_5_jogos":       form5,
        "pontos_forma_5":      pts_form5,
        "ultimo_jogo":         hist.iloc[-1]["date"].strftime("%Y-%m-%d"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tool 2 — get_h2h
# ══════════════════════════════════════════════════════════════════════════════
def get_h2h(team_a: str, team_b: str, last_n: int = 10) -> dict:
    """
    Histórico de confrontos diretos: vitórias, empates, médias de
    gols, cartões e escanteios por jogo entre os dois times.
    """
    df   = data_loader.full()
    mask = (
        ((df["home_team"] == team_a) & (df["away_team"] == team_b)) |
        ((df["home_team"] == team_b) & (df["away_team"] == team_a))
    )
    hist = df[mask].sort_values("date").tail(last_n)

    if len(hist) == 0:
        return {"error": f"Nenhum confronto encontrado entre '{team_a}' e '{team_b}'."}

    wins_a = sum(
        1 for _, r in hist.iterrows()
        if (r["home_team"] == team_a and r["result"] == "H") or
           (r["away_team"] == team_a and r["result"] == "A")
    )
    draws    = int((hist["result"] == "D").sum())
    wins_b   = len(hist) - wins_a - draws
    last_row = hist.iloc[-1]

    return {
        "time_a":           team_a,
        "time_b":           team_b,
        "confrontos":       len(hist),
        "vitorias_a":       wins_a,
        "empates":          draws,
        "vitorias_b":       wins_b,
        "gols_por_jogo":    round(float((hist["home_score"] + hist["away_score"]).mean()), 2),
        "amarelos_avg":     round(float(hist["total_yellow_cards"].mean()), 2) if "total_yellow_cards" in hist.columns else None,
        "escanteios_avg":   round(float(hist["total_corners"].mean()), 2) if "total_corners" in hist.columns else None,
        "ultimo_confronto": last_row["date"].strftime("%Y-%m-%d"),
        "resultado_ultimo": f"{last_row['home_team']} {last_row['home_score']}×{last_row['away_score']} {last_row['away_team']}",
    }


# ══════════════════════════════════════════════════════════════════════════════
# Tool 3 — get_league_table
# ══════════════════════════════════════════════════════════════════════════════
def get_league_table(league: str, season: str = "current") -> list:
    """
    Tabela classificatória: posição, pontos, saldo de gols,
    aproveitamento, gols marcados e sofridos.
    """
    m = data_loader.matches()

    if season == "current":
        seasons_avail = m[m["league"] == league]["season"].unique()
        if len(seasons_avail) == 0:
            return [{"error": f"Liga '{league}' não encontrada."}]
        season = sorted(seasons_avail)[-1]

    hist = m[(m["league"] == league) & (m["season"] == season)]
    if len(hist) == 0:
        return [{"error": f"Dados não encontrados para {league} temporada {season}."}]

    teams: dict = {}
    for _, r in hist.iterrows():
        for t in [r["home_team"], r["away_team"]]:
            if t not in teams:
                teams[t] = {"time": t, "P": 0, "J": 0, "V": 0, "E": 0, "D": 0, "GM": 0, "GS": 0}

        h, a = r["home_team"], r["away_team"]
        teams[h]["J"] += 1; teams[a]["J"] += 1
        teams[h]["GM"] += r["home_score"]; teams[h]["GS"] += r["away_score"]
        teams[a]["GM"] += r["away_score"]; teams[a]["GS"] += r["home_score"]

        if r["result"] == "H":
            teams[h]["V"] += 1; teams[h]["P"] += 3; teams[a]["D"] += 1
        elif r["result"] == "D":
            teams[h]["E"] += 1; teams[h]["P"] += 1
            teams[a]["E"] += 1; teams[a]["P"] += 1
        else:
            teams[a]["V"] += 1; teams[a]["P"] += 3; teams[h]["D"] += 1

    table = sorted(teams.values(), key=lambda x: (x["P"], x["V"], x["GM"] - x["GS"]), reverse=True)
    for i, row in enumerate(table):
        row["pos"]   = i + 1
        row["SG"]    = row["GM"] - row["GS"]
        row["aprov"] = round(row["P"] / max(1, row["J"] * 3) * 100, 1)

    return table


# ══════════════════════════════════════════════════════════════════════════════
# Tool 4 — get_team_schedule
# ══════════════════════════════════════════════════════════════════════════════
def get_team_schedule(team_name: str, upcoming: bool = False) -> dict:
    """
    Próximos jogos ou últimas partidas de um time.
    """
    m    = data_loader.matches()
    mask = (m["home_team"] == team_name) | (m["away_team"] == team_name)
    sub  = m[mask].sort_values("date")

    if len(sub) == 0:
        return {"error": f"Time '{team_name}' não encontrado."}

    today = pd.Timestamp.now()
    if upcoming:
        games = sub[sub["date"] > today].head(5)
        key   = "proximos_jogos"
    else:
        games = sub[sub["date"] <= today].tail(5)
        key   = "ultimos_jogos"

    records = []
    for _, r in games.iterrows():
        is_home = r["home_team"] == team_name
        rec = {
            "data":  r["date"].strftime("%Y-%m-%d"),
            "liga":  r["league"],
            "casa":  r["home_team"],
            "fora":  r["away_team"],
            "mando": "casa" if is_home else "fora",
        }
        if not upcoming:
            rec["placar"]    = f"{r['home_score']}×{r['away_score']}"
            rec["resultado"] = _result_for_team(r, team_name)
        records.append(rec)

    return {"time": team_name, key: records}


# ══════════════════════════════════════════════════════════════════════════════
# Tool 5 — run_prediction_engine
# ══════════════════════════════════════════════════════════════════════════════
def run_prediction_engine(home_team: str, away_team: str, league: str) -> dict:
    """
    Aciona a rede neural para prever resultado, cartões e escanteios
    de uma partida futura.
    """
    return pred_module.predict(home_team, away_team, league)


# ══════════════════════════════════════════════════════════════════════════════
# Dispatcher — usado pelo orchestrator.py
# ══════════════════════════════════════════════════════════════════════════════
TOOL_MAP = {
    "get_team_stats":        get_team_stats,
    "get_h2h":               get_h2h,
    "get_league_table":      get_league_table,
    "get_team_schedule":     get_team_schedule,
    "run_prediction_engine": run_prediction_engine,
}

def execute_tool(name: str, inputs: dict):
    """Executa uma tool pelo nome. Retorna dict com resultado ou erro."""
    if name not in TOOL_MAP:
        return {"error": f"Tool '{name}' não encontrada. Disponíveis: {list(TOOL_MAP.keys())}"}
    try:
        return TOOL_MAP[name](**inputs)
    except Exception as e:
        return {"error": f"Erro ao executar '{name}': {str(e)}"}


# ══════════════════════════════════════════════════════════════════════════════
# Schemas para a Claude API (Tool Use)
# ══════════════════════════════════════════════════════════════════════════════
TOOL_DEFINITIONS = [
    {
        "name": "get_team_stats",
        "description": (
            "Retorna estatísticas recentes de um time: aproveitamento, gols marcados/sofridos, "
            "médias de cartões amarelos, escanteios, posse e forma dos últimos N jogos. "
            "Use sempre que o usuário perguntar sobre desempenho, fase ou forma de um time."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string",  "description": "Nome exato do time"},
                "last_n":    {"type": "integer", "description": "Últimos N jogos (padrão: 10)"},
                "league":    {"type": "string",  "description": "Filtro de liga: 'brasileirao_a', 'premier_league' ou 'la_liga' (opcional)"},
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "get_h2h",
        "description": (
            "Retorna o histórico de confrontos diretos entre dois times: vitórias de cada lado, "
            "empates, médias de gols, cartões e escanteios. "
            "Use quando o usuário comparar dois times ou perguntar sobre confrontos anteriores."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_a": {"type": "string",  "description": "Nome do primeiro time"},
                "team_b": {"type": "string",  "description": "Nome do segundo time"},
                "last_n": {"type": "integer", "description": "Últimos N confrontos (padrão: 10)"},
            },
            "required": ["team_a", "team_b"],
        },
    },
    {
        "name": "get_league_table",
        "description": (
            "Retorna a tabela classificatória completa de uma liga com posição, pontos, "
            "saldo de gols e aproveitamento. Use quando o usuário perguntar sobre classificação, "
            "líderes ou posição de um time na tabela."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "league": {"type": "string", "description": "Liga: 'brasileirao_a', 'premier_league' ou 'la_liga'"},
                "season": {"type": "string", "description": "Temporada ex: '24/25'. Use 'current' para a mais recente (padrão)."},
            },
            "required": ["league"],
        },
    },
    {
        "name": "get_team_schedule",
        "description": (
            "Retorna os últimos jogos realizados ou os próximos jogos de um time, "
            "com data, adversário, mando de campo e resultado. "
            "Use quando o usuário perguntar sobre agenda ou jogos recentes."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "team_name": {"type": "string",  "description": "Nome do time"},
                "upcoming":  {"type": "boolean", "description": "true para próximos jogos, false para jogos recentes (padrão: false)"},
            },
            "required": ["team_name"],
        },
    },
    {
        "name": "run_prediction_engine",
        "description": (
            "Aciona a rede neural treinada para prever o resultado, total de cartões amarelos "
            "e total de escanteios de uma partida FUTURA. "
            "Use SEMPRE que o usuário mencionar dois times em contexto de jogo futuro — "
            "mesmo implícito: 'quem vence', 'vai ter gol', 'apostaria em quê', 'quem favorito'. "
            "NÃO use para jogos já realizados."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "home_team": {"type": "string", "description": "Nome do time mandante"},
                "away_team": {"type": "string", "description": "Nome do time visitante"},
                "league":    {"type": "string", "description": "Liga: 'brasileirao_a', 'premier_league' ou 'la_liga'"},
            },
            "required": ["home_team", "away_team", "league"],
        },
    },
]
