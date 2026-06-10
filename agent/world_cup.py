# -*- coding: utf-8 -*-
"""
world_cup.py — previsões da COPA DO MUNDO 2026 usando a forma dos AMISTOSOS.

Busca os jogos da Copa (Sofascore, unique-tournament 16, season 2026) e roda a
rede neural para cada confronto, usando as features das seleções obtidas dos
amistosos (liga 'amistosos' no banco). Se houver odds, também calcula apostas de valor.

⚠️ IMPORTANTE: o modelo foi treinado em futebol de CLUBES. Previsões de seleções
são uma EXTRAPOLAÇÃO (out-of-distribution) — as features são genéricas (forma, xG,
posse...), mas o modelo nunca viu seleções. Trate como experimental.

Pré-requisito: ter coletado os amistosos e re-preprocessado:
    python update_weekly.py --no-retrain --seasons 2

Uso:
    python agent/world_cup.py                 # lista os próximos jogos da Copa
    python agent/world_cup.py --floor 0.05    # inclui apostas de valor (se houver odds)
    python agent/world_cup.py --max 30
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import argparse
import datetime
from pathlib import Path

_AGENT = Path(__file__).resolve().parent
sys.path.insert(0, str(_AGENT / "src"))
sys.path.insert(0, str(_AGENT))
import predictor as pred_module                  # noqa: E402 (torch primeiro)
import value_bets as vb                          # noqa: E402

WC_TOURNAMENT = 16
WC_SEASON = 58210          # Copa do Mundo 2026
FORM_LEAGUE = "amistosos"  # forma das seleções vem dos amistosos
RES_NAME = {"H": "Mandante", "D": "Empate", "A": "Visitante"}


def wc_fixtures(cli, max_matches=30):
    d = cli.get(f"{vb.BASE}/unique-tournament/{WC_TOURNAMENT}/season/{WC_SEASON}/events/next/0")
    out = []
    for e in (d or {}).get("events", [])[:max_matches]:
        out.append({"id": e["id"], "home": e["homeTeam"]["name"],
                    "away": e["awayTeam"]["name"], "ts": e.get("startTimestamp"),
                    "round": e.get("roundInfo", {}).get("name") or
                             (e.get("tournament", {}).get("name", ""))})
    return out


def predict_wc(home, away, floor=None):
    """Previsão de um confronto da Copa (forma dos amistosos) + apostas de valor."""
    pred = pred_module.predict(home, away, FORM_LEAGUE)
    if "error" in pred:
        return pred
    pred["_bets"] = []
    return pred


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=None, help="piso de EV p/ apostas (ex.: 0.05)")
    ap.add_argument("--max", type=int, default=30)
    args = ap.parse_args()

    cli = vb.Odds()
    fixtures = wc_fixtures(cli, args.max)
    print(f"Copa do Mundo 2026 — {len(fixtures)} jogos encontrados.")
    print("⚠️  Modelo treinado em clubes; previsões de seleções são extrapolação.\n")
    if not fixtures:
        print("(Sem jogos no fixture — verifique a temporada/torneio.)")
        return

    print(f"{'Jogo':<34}{'Data':>12}  {'Palpite':<10}{'P(H/D/A)':>16}")
    print("-" * 80)
    achou = 0
    naofachou = []
    for fx in fixtures:
        pred = pred_module.predict(fx["home"], fx["away"], FORM_LEAGUE)
        data = datetime.datetime.utcfromtimestamp(fx["ts"]).strftime("%d/%m %H:%M") if fx["ts"] else ""
        if "error" in pred:
            naofachou = naofachou
            naofachou.append(f"{fx['home']} x {fx['away']}")
            continue
        achou += 1
        r = pred["resultado"]
        pr = r["probs"]
        eq = " [equilibrado]" if r.get("equilibrado") else ""
        pstr = f"H{pr['H']*100:3.0f} D{pr['D']*100:3.0f} A{pr['A']*100:3.0f}"
        print(f"{fx['home']+' x '+fx['away']:<34}{data:>12}  {RES_NAME[r['label']]:<10}{pstr:>16}{eq}")
        # apostas de valor (se odds disponíveis)
        if args.floor is not None:
            odds = vb.parse_odds(cli.get(f"{vb.BASE}/event/{fx['id']}/odds/1/all"))
            if odds:
                for b in vb.value_bets_for_match(fx["home"], fx["away"], FORM_LEAGUE, odds, args.floor):
                    if "erro" not in b:
                        print(f"      💰 {b['mercado']}: {b['aposta']} @ {b['odd']}  EV +{b['ev_pct']}%  (stake {b['stake_pct']}%)")

    print("-" * 80)
    print(f"Previstos: {achou}/{len(fixtures)}")
    if naofachou:
        print(f"Sem forma de amistosos (não previstos): {', '.join(naofachou[:12])}"
              + (" …" if len(naofachou) > 12 else ""))


if __name__ == "__main__":
    main()
