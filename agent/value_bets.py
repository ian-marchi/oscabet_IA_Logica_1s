# -*- coding: utf-8 -*-
"""
value_bets.py — apostas de VALOR em jogos futuros (modelo + odds reais).

Busca jogos futuros no Sofascore + suas odds, roda a rede neural e recomenda
apostas onde o valor esperado (EV = prob_modelo × odd − 1) passa de um PISO.

Mercados cobertos (a odd precisa estar na MESMA linha do modelo):
  - Resultado (1X2)         → odds "Full time"
  - Escanteios Over/Under 9.5 → odds "Corners 2-Way" linha 9.5
  - Cartões Over/Under 4.5    → odds "Cards in match" linha 4.5 (só se a casa oferecer 4.5)

Uso (no ambiente conda 'oscabet'):
    python agent/value_bets.py                          # todas as ligas, piso 5%
    python agent/value_bets.py --league brasileirao_a   # só uma liga
    python agent/value_bets.py --floor 0.08 --max 8     # piso 8%, até 8 jogos/liga
    python agent/value_bets.py --event 15235586         # avalia um match_id específico (teste)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import time
import random
import argparse
from pathlib import Path

# torch ANTES de pandas (Windows): predictor importa torch primeiro.
_SRC = Path(__file__).resolve().parent / "src"
sys.path.insert(0, str(_SRC))
import predictor as pred_module                          # noqa: E402

import cloudscraper                                      # noqa: E402

BASE = "https://api.sofascore.com/api/v1"
CARD_LINE = 4.5
CORNER_LINE = 9.5
DEFAULT_FLOOR = 0.05      # piso de EV (5%)
KELLY_FRACTION = 0.25     # fração de Kelly p/ sugestão de stake
KELLY_CAP = 0.05          # stake máx sugerido = 5% da banca

# Ligas (chave interna → id do torneio no Sofascore) — espelha o scraper
LEAGUE_IDS = {
    "brasileirao_a": 325, "brasileirao_b": 390, "copa_brasil": 162,
    "libertadores": 384, "premier_league": 17, "la_liga": 8,
    "serie_a": 23, "bundesliga": 35, "ligue_1": 34, "champions_league": 7,
}


class Odds:
    """Cliente leve p/ a API do Sofascore (delay curto; uso sob demanda)."""
    def __init__(self):
        self.s = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False})
        self.s.headers.update({"Referer": "https://www.sofascore.com/",
                               "Origin": "https://www.sofascore.com"})

    def get(self, url):
        try:
            r = self.s.get(url, timeout=25)
            time.sleep(random.uniform(0.6, 1.3))
            return r.json() if r.status_code == 200 else None
        except Exception:
            return None

    def current_season(self, tid):
        d = self.get(f"{BASE}/unique-tournament/{tid}/seasons")
        ss = (d or {}).get("seasons", [])
        return ss[0]["id"] if ss else None

    def upcoming(self, league_key, max_matches=10):
        tid = LEAGUE_IDS[league_key]
        sid = self.current_season(tid)
        if not sid:
            return []
        d = self.get(f"{BASE}/unique-tournament/{tid}/season/{sid}/events/next/0")
        out = []
        for e in (d or {}).get("events", [])[:max_matches]:
            if e.get("status", {}).get("type") == "finished":
                continue
            out.append({"id": e["id"], "home": e["homeTeam"]["name"],
                        "away": e["awayTeam"]["name"], "league": league_key,
                        "ts": e.get("startTimestamp")})
        return out


def _dec(frac: str):
    """Converte odd fracional 'n/d' em decimal (n/d + 1)."""
    try:
        n, d = frac.split("/")
        return round(int(n) / int(d) + 1, 3)
    except Exception:
        return None


def parse_odds(raw: dict) -> dict:
    """Extrai 1X2, escanteios e cartões (com a linha) das odds cruas."""
    res = {}
    for m in (raw or {}).get("markets", []):
        name = m.get("marketName", "")
        ch = {c.get("name"): _dec(c.get("fractionalValue", "")) for c in m.get("choices", [])}
        if name == "Full time" and {"1", "X", "2"} <= set(ch):
            res["1x2"] = {"H": ch["1"], "D": ch["X"], "A": ch["2"]}
        elif name == "Corners 2-Way":
            line = m.get("choiceGroup")
            res["corners"] = {"line": float(line) if line else None,
                              "Over": ch.get("Over"), "Under": ch.get("Under")}
        elif name == "Cards in match":
            line = m.get("choiceGroup")
            res["cards"] = {"line": float(line) if line else None,
                            "Over": ch.get("Over"), "Under": ch.get("Under")}
    return res


def _kelly_stake(p, o):
    """Stake sugerido = fração de Kelly (capada), em % da banca."""
    edge = p * o - 1
    if edge <= 0 or o <= 1:
        return 0.0
    kelly = edge / (o - 1)
    return round(min(kelly * KELLY_FRACTION, KELLY_CAP) * 100, 2)


def value_bets_for_match(home, away, league, odds, floor) -> list:
    """Compara probs do modelo com as odds e devolve apostas com EV ≥ piso."""
    pred = pred_module.predict(home, away, league)
    if "error" in pred:
        return [{"erro": pred["error"]}]
    bets = []

    def consider(market, label, p, o):
        if o is None or p is None:
            return
        ev = p * o - 1
        if ev >= floor:
            bets.append({"mercado": market, "aposta": label, "prob": round(p, 3),
                         "odd": o, "ev_pct": round(ev * 100, 1), "stake_pct": _kelly_stake(p, o)})

    # Resultado (1X2)
    if "1x2" in odds:
        pr = pred["resultado"]["probs"]
        consider("Resultado", "Casa (1)",      pr["H"], odds["1x2"]["H"])
        consider("Resultado", "Empate (X)",    pr["D"], odds["1x2"]["D"])
        consider("Resultado", "Visitante (2)", pr["A"], odds["1x2"]["A"])

    # Escanteios — só se a linha bater (9.5)
    c = odds.get("corners")
    if c and c.get("line") == CORNER_LINE:
        pc = pred["escanteios"]["probs"]
        consider("Escanteios", "Over 9.5",  pc["Over 9.5"],  c["Over"])
        consider("Escanteios", "Under 9.5", pc["Under 9.5"], c["Under"])

    # Cartões — só se a casa oferecer a linha 4.5
    k = odds.get("cards")
    if k and k.get("line") == CARD_LINE:
        py = pred["cartoes"]["probs"]
        consider("Cartões", "Over 4.5",  py["Over 4.5"],  k["Over"])
        consider("Cartões", "Under 4.5", py["Under 4.5"], k["Under"])

    return bets


def collect(leagues, floor=DEFAULT_FLOOR, max_per_league=10):
    """Varre jogos futuros das ligas e devolve as apostas de valor encontradas."""
    cli = Odds()
    rows = []
    sem_odds = 0
    for lg in leagues:
        for fx in cli.upcoming(lg, max_per_league):
            raw = cli.get(f"{BASE}/event/{fx['id']}/odds/1/all")
            odds = parse_odds(raw)
            if not odds:
                sem_odds += 1
                continue
            for b in value_bets_for_match(fx["home"], fx["away"], lg, odds, floor):
                if "erro" in b:
                    continue
                b.update({"jogo": f"{fx['home']} x {fx['away']}", "liga": lg, "id": fx["id"]})
                rows.append(b)
    rows.sort(key=lambda r: r["ev_pct"], reverse=True)
    return rows, sem_odds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--league", default=None, help="liga única (default: todas)")
    ap.add_argument("--floor", type=float, default=DEFAULT_FLOOR, help="piso de EV (ex.: 0.05 = 5%)")
    ap.add_argument("--max", type=int, default=10, help="máx. de jogos por liga")
    ap.add_argument("--event", type=int, default=None, help="avalia um match_id específico (teste)")
    args = ap.parse_args()

    if args.event:
        cli = Odds()
        ev = cli.get(f"{BASE}/event/{args.event}")
        e = (ev or {}).get("event", {})
        home, away = e.get("homeTeam", {}).get("name"), e.get("awayTeam", {}).get("name")
        league = next((k for k in LEAGUE_IDS), "brasileirao_a")
        # tenta inferir a liga pelo torneio do evento
        tid = e.get("tournament", {}).get("uniqueTournament", {}).get("id")
        for k, v in LEAGUE_IDS.items():
            if v == tid:
                league = k
        odds = parse_odds(cli.get(f"{BASE}/event/{args.event}/odds/1/all"))
        print(f"Jogo: {home} x {away} ({league}) | odds: {list(odds.keys())}")
        for b in value_bets_for_match(home, away, league, odds, args.floor):
            print(" ", b)
        return

    leagues = [args.league] if args.league else list(LEAGUE_IDS)
    print(f"Buscando apostas de valor (piso {args.floor*100:.0f}%) em {len(leagues)} liga(s)…")
    rows, sem_odds = collect(leagues, args.floor, args.max)
    print(f"\n{len(rows)} apostas de valor encontradas ({sem_odds} jogos sem odds disponíveis).\n")
    if rows:
        print(f"{'Jogo':<34}{'Liga':<16}{'Mercado':<12}{'Aposta':<14}{'Odd':>6}{'Prob':>7}{'EV':>7}{'Stake':>7}")
        print("-" * 110)
        for r in rows:
            print(f"{r['jogo'][:33]:<34}{r['liga']:<16}{r['mercado']:<12}{r['aposta']:<14}"
                  f"{r['odd']:>6}{r['prob']*100:>6.0f}%{r['ev_pct']:>6.1f}%{r['stake_pct']:>6.1f}%")
    else:
        print("(Nenhuma aposta de valor agora — pode ser período sem jogos próximos com odds.)")


if __name__ == "__main__":
    main()
