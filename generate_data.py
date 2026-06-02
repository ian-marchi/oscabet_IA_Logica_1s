#!/usr/bin/env python3
"""
OscaBet — Gerador de Dados Sintéticos
======================================
Gera data/raw/matches.csv e data/raw/match_stats.csv com o
mesmo schema que o scraper real vai produzir.

Para substituir pelos dados reais, basta trocar os CSVs em data/raw/
e executar o notebook 02_preprocessamento novamente.
"""
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

np.random.seed(42)
random.seed(42)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(BASE_DIR, "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

# ─── Configuração de ligas, times e estilos de jogo ──────────────────────────
LEAGUES = {
    "brasileirao_a": {
        "name": "Brasileirão Série A",
        "seasons": {
            "20/21": "2020-02-01",
            "21/22": "2021-05-30",
            "22/23": "2022-04-10",
            "23/24": "2023-04-15",
            "24/25": "2024-04-13",
        },
        "teams": [
            ("Flamengo", 1001, 0.72), ("Palmeiras", 1002, 0.71),
            ("Corinthians", 1003, 0.58), ("São Paulo", 1004, 0.57),
            ("Grêmio", 1005, 0.59), ("Internacional", 1006, 0.60),
            ("Atlético-MG", 1007, 0.68), ("Cruzeiro", 1008, 0.55),
            ("Santos", 1009, 0.56), ("Vasco", 1010, 0.50),
            ("Fluminense", 1011, 0.56), ("Botafogo", 1012, 0.55),
            ("Athletico-PR", 1013, 0.57), ("Bragantino", 1014, 0.56),
            ("Fortaleza", 1015, 0.55), ("Ceará", 1016, 0.48),
            ("Sport", 1017, 0.44), ("América-MG", 1018, 0.46),
            ("Cuiabá", 1019, 0.43), ("Goiás", 1020, 0.44),
        ],
        "style": {
            "corners_lambda": 5.2, "yellow_home": 2.6, "yellow_away": 2.9,
            "shots_home": 13.0, "shots_away": 10.5, "poss_home_mean": 52.0,
        },
    },
    "premier_league": {
        "name": "Premier League",
        "seasons": {
            "20/21": "2020-09-12", "21/22": "2021-08-13",
            "22/23": "2022-08-05", "23/24": "2023-08-11", "24/25": "2024-08-16",
        },
        "teams": [
            ("Manchester City", 2001, 0.76), ("Liverpool", 2002, 0.72),
            ("Chelsea", 2003, 0.65), ("Arsenal", 2004, 0.64),
            ("Manchester United", 2005, 0.62), ("Tottenham", 2006, 0.61),
            ("Leicester", 2007, 0.57), ("West Ham", 2008, 0.55),
            ("Everton", 2009, 0.50), ("Wolverhampton", 2010, 0.51),
            ("Newcastle", 2011, 0.52), ("Aston Villa", 2012, 0.54),
            ("Crystal Palace", 2013, 0.47), ("Southampton", 2014, 0.46),
            ("Brighton", 2015, 0.51), ("Brentford", 2016, 0.50),
            ("Fulham", 2017, 0.46), ("Burnley", 2018, 0.44),
            ("Leeds", 2019, 0.46), ("Nottingham Forest", 2020, 0.45),
        ],
        "style": {
            "corners_lambda": 5.0, "yellow_home": 1.9, "yellow_away": 2.1,
            "shots_home": 13.5, "shots_away": 11.0, "poss_home_mean": 53.0,
        },
    },
    "la_liga": {
        "name": "La Liga",
        "seasons": {
            "20/21": "2020-09-12", "21/22": "2021-08-13",
            "22/23": "2022-08-12", "23/24": "2023-08-11", "24/25": "2024-08-16",
        },
        "teams": [
            ("Real Madrid", 3001, 0.76), ("Barcelona", 3002, 0.73),
            ("Atletico Madrid", 3003, 0.67), ("Sevilla", 3004, 0.60),
            ("Real Betis", 3005, 0.55), ("Valencia", 3006, 0.53),
            ("Villarreal", 3007, 0.57), ("Athletic Bilbao", 3008, 0.55),
            ("Osasuna", 3009, 0.48), ("Celta Vigo", 3010, 0.49),
            ("Real Sociedad", 3011, 0.56), ("Getafe", 3012, 0.47),
            ("Levante", 3013, 0.44), ("Elche", 3014, 0.43),
            ("Mallorca", 3015, 0.46), ("Cadiz", 3016, 0.41),
            ("Alaves", 3017, 0.43), ("Valladolid", 3018, 0.42),
            ("Rayo Vallecano", 3019, 0.47), ("Espanyol", 3020, 0.46),
        ],
        "style": {
            "corners_lambda": 4.8, "yellow_home": 2.4, "yellow_away": 2.7,
            "shots_home": 12.5, "shots_away": 10.0, "poss_home_mean": 54.0,
        },
    },
}

# ─── Gerador de estatísticas por partida ─────────────────────────────────────
def gen_stats(style: dict, home_str: float, away_str: float) -> dict:
    ratio = home_str / (home_str + away_str)  # [0,1]; >0.5 = home é mais forte

    # Posse
    poss_h = float(np.clip(np.random.normal(style["poss_home_mean"] + (ratio - 0.5) * 14, 7), 28, 72))
    poss_a = round(100 - poss_h, 1)
    poss_h = round(poss_h, 1)

    # xG (motor do placar)
    xg_h = max(0.1, np.random.normal(style["shots_home"] * 0.1 * (0.6 + ratio * 0.8), 0.45))
    xg_a = max(0.1, np.random.normal(style["shots_away"] * 0.1 * (0.6 + (1 - ratio) * 0.8), 0.40))

    # Chutes
    sh_h = max(1, int(np.random.normal(style["shots_home"] * (0.65 + ratio * 0.70), 3.5)))
    sh_a = max(1, int(np.random.normal(style["shots_away"] * (0.65 + (1-ratio) * 0.70), 3.0)))

    sot_h = max(0, min(sh_h, int(np.random.binomial(sh_h, 0.37))))
    sot_a = max(0, min(sh_a, int(np.random.binomial(sh_a, 0.34))))
    rem_h, rem_a = sh_h - sot_h, sh_a - sot_a
    sblk_h = max(0, min(rem_h, int(np.random.binomial(rem_h, 0.33))))
    sblk_a = max(0, min(rem_a, int(np.random.binomial(rem_a, 0.33))))
    soff_h, soff_a = max(0, rem_h - sblk_h), max(0, rem_a - sblk_a)

    # Grandes chances
    bc_h = max(0, int(np.random.poisson(max(0.3, sot_h * 0.50))))
    bc_a = max(0, int(np.random.poisson(max(0.3, sot_a * 0.45))))
    bcm_h = max(0, int(np.random.binomial(bc_h, 0.38)))
    bcm_a = max(0, int(np.random.binomial(bc_a, 0.38)))

    # Escanteios
    c_lam = style["corners_lambda"]
    corn_h = max(0, int(np.random.poisson(c_lam * (0.65 + ratio * 0.70))))
    corn_a = max(0, int(np.random.poisson(c_lam * (0.65 + (1-ratio) * 0.70))))

    # Cartões
    yell_h = max(0, int(np.random.poisson(style["yellow_home"])))
    yell_a = max(0, int(np.random.poisson(style["yellow_away"])))
    red_h  = 1 if np.random.random() < 0.025 else 0
    red_a  = 1 if np.random.random() < 0.035 else 0

    # Faltas e impedimentos
    fouls_h = max(3, int(np.random.normal(12.5, 3)))
    fouls_a = max(3, int(np.random.normal(13.5, 3)))
    offs_h  = max(0, int(np.random.poisson(2.0)))
    offs_a  = max(0, int(np.random.poisson(1.8)))

    # Passes
    pass_h     = max(100, int(poss_h * 9 + np.random.normal(0, 40)))
    pass_a     = max(100, int(poss_a * 9 + np.random.normal(0, 40)))
    pass_acc_h = max(0, min(pass_h, int(pass_h * float(np.clip(np.random.normal(0.83, 0.05), 0.55, 0.96)))))
    pass_acc_a = max(0, min(pass_a, int(pass_a * float(np.clip(np.random.normal(0.81, 0.05), 0.55, 0.96)))))

    # Duelos
    tack_h  = max(5, int(np.random.normal(17, 5)))
    tack_a  = max(5, int(np.random.normal(19, 5)))
    inter_h = max(2, int(np.random.normal(11, 4)))
    inter_a = max(2, int(np.random.normal(12, 4)))
    clear_h = max(3, int(np.random.normal(18 + (1-ratio)*10, 6)))
    clear_a = max(3, int(np.random.normal(22 + ratio*10, 6)))

    # Aéreos
    tot_aer   = max(15, int(np.random.normal(44, 11)))
    aer_tot_h = max(5, tot_aer // 2)
    aer_tot_a = max(5, tot_aer - aer_tot_h)
    aer_win_h = max(0, min(aer_tot_h, int(np.random.binomial(aer_tot_h, 0.52 + (ratio-0.5)*0.10))))
    aer_win_a = max(0, min(aer_tot_a, int(np.random.binomial(aer_tot_a, 0.48 + (0.5-ratio)*0.10))))

    # Dribles e defesas
    drib_h  = max(0, int(np.random.poisson(5)))
    drib_a  = max(0, int(np.random.poisson(4)))
    saves_h = max(0, sot_a - max(0, int(np.random.poisson(xg_a * 0.85))))
    saves_a = max(0, sot_h - max(0, int(np.random.poisson(xg_h * 0.90))))

    return dict(
        home_possession=poss_h, away_possession=poss_a,
        home_shots=sh_h, away_shots=sh_a,
        home_shots_on_target=sot_h, away_shots_on_target=sot_a,
        home_shots_off_target=soff_h, away_shots_off_target=soff_a,
        home_shots_blocked=sblk_h, away_shots_blocked=sblk_a,
        home_big_chances=bc_h, away_big_chances=bc_a,
        home_big_chances_missed=bcm_h, away_big_chances_missed=bcm_a,
        home_corners=corn_h, away_corners=corn_a, total_corners=corn_h+corn_a,
        home_yellow_cards=yell_h, away_yellow_cards=yell_a, total_yellow_cards=yell_h+yell_a,
        home_red_cards=red_h, away_red_cards=red_a,
        home_fouls=fouls_h, away_fouls=fouls_a,
        home_offsides=offs_h, away_offsides=offs_a,
        home_passes=pass_h, away_passes=pass_a,
        home_passes_accurate=pass_acc_h, away_passes_accurate=pass_acc_a,
        home_tackles=tack_h, away_tackles=tack_a,
        home_interceptions=inter_h, away_interceptions=inter_a,
        home_clearances=clear_h, away_clearances=clear_a,
        home_aerial_duels_won=aer_win_h, away_aerial_duels_won=aer_win_a,
        home_aerial_duels_total=aer_tot_h, away_aerial_duels_total=aer_tot_a,
        home_dribbles_successful=drib_h, away_dribbles_successful=drib_a,
        home_saves=saves_h, away_saves=saves_a,
        home_xg=round(xg_h, 2), away_xg=round(xg_a, 2),
    )

# ─── Loop principal de geração ────────────────────────────────────────────────
def generate():
    matches_rows, stats_rows = [], []
    match_id = 1

    for league_key, cfg in LEAGUES.items():
        teams  = cfg["teams"]   # list of (name, id, strength)
        style  = cfg["style"]
        n_teams = len(teams)

        for season, start_str in cfg["seasons"].items():
            start_dt = datetime.strptime(start_str, "%Y-%m-%d")

            # Double round-robin
            pairings = [(teams[i], teams[j]) for i in range(n_teams) for j in range(n_teams) if i != j]
            random.shuffle(pairings)

            for idx, (home_t, away_t) in enumerate(pairings):
                match_date = start_dt + timedelta(days=idx * 7 // 10)
                h_name, h_id, h_str = home_t
                a_name, a_id, a_str = away_t

                stats = gen_stats(style, h_str, a_str)

                h_goals = max(0, int(np.random.poisson(max(0.1, stats["home_xg"]))))
                a_goals = max(0, int(np.random.poisson(max(0.1, stats["away_xg"]))))
                result  = "H" if h_goals > a_goals else ("A" if a_goals > h_goals else "D")

                has_stats = np.random.random() > 0.05   # ~5% sem stats

                matches_rows.append(dict(
                    match_id=match_id, league=league_key, season=season,
                    date=match_date.strftime("%Y-%m-%d"),
                    home_team=h_name, away_team=a_name,
                    home_team_id=h_id, away_team_id=a_id,
                    home_score=h_goals, away_score=a_goals,
                    result=result, stats_collected=has_stats,
                ))
                if has_stats:
                    stats_rows.append(dict(match_id=match_id, **stats))

                match_id += 1

    matches = pd.DataFrame(matches_rows)
    stats   = pd.DataFrame(stats_rows)

    matches["date"] = pd.to_datetime(matches["date"])
    matches = matches.sort_values("date").reset_index(drop=True)

    matches.to_csv(os.path.join(RAW_DIR, "matches.csv"),     index=False)
    stats.to_csv(  os.path.join(RAW_DIR, "match_stats.csv"), index=False)

    print(f"✅ matches.csv      → {len(matches):,} partidas")
    print(f"✅ match_stats.csv  → {len(stats):,} partidas com estatísticas")
    ligas = matches["league"].value_counts().to_dict()
    for k, v in ligas.items():
        print(f"   {k}: {v:,} jogos")

if __name__ == "__main__":
    generate()
