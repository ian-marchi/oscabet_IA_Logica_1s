# -*- coding: utf-8 -*-
"""
test_suite.py — testes automatizados (com asserções) do núcleo do OscaBet.

Cobre: predictor (estrutura, linhas 4.5/9.5, empate/equilibrado, cross-league,
erro), motor de apostas de valor (frações→decimal, parsing, EV, casamento de
linha) e o dispatcher de tools. Roda sem internet e sem a LLM.

Uso (no ambiente conda 'oscabet'):
    python agent/src/test_suite.py
    # ou com pytest, se preferir:  pytest agent/src/test_suite.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import traceback
from pathlib import Path

_SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(_SRC))
sys.path.insert(0, str(_SRC.parent))  # agent/ (para value_bets)

import predictor                              # noqa: E402 (importa torch primeiro)
import tools                                  # noqa: E402
import value_bets as vb                       # noqa: E402


# ─────────────────────────── value_bets (matemática) ─────────────────────────
def test_decimal_from_fractional():
    assert vb._dec("43/50") == 1.86          # 0.86 + 1
    assert vb._dec("10/3") == round(10/3 + 1, 3)
    assert vb._dec("nao-fracao") is None


def test_parse_odds_lines():
    raw = {"markets": [
        {"marketName": "Full time", "choices": [
            {"name": "1", "fractionalValue": "1/1"},
            {"name": "X", "fractionalValue": "2/1"},
            {"name": "2", "fractionalValue": "3/1"}]},
        {"marketName": "Corners 2-Way", "choiceGroup": "9.5", "choices": [
            {"name": "Over", "fractionalValue": "1/1"},
            {"name": "Under", "fractionalValue": "1/1"}]},
        {"marketName": "Cards in match", "choiceGroup": "5.5", "choices": [
            {"name": "Over", "fractionalValue": "1/1"},
            {"name": "Under", "fractionalValue": "1/1"}]},
    ]}
    o = vb.parse_odds(raw)
    assert o["1x2"] == {"H": 2.0, "D": 3.0, "A": 4.0}
    assert o["corners"]["line"] == 9.5
    assert o["cards"]["line"] == 5.5


def test_value_bets_skips_wrong_card_line():
    # cartões na linha 5.5 (≠ 4.5 do modelo) → NÃO deve recomendar cartões.
    odds = {"1x2": {"H": 10, "D": 10, "A": 10},
            "corners": {"line": 9.5, "Over": 10, "Under": 10},
            "cards": {"line": 5.5, "Over": 10, "Under": 10}}
    bets = vb.value_bets_for_match("Flamengo", "Grêmio", "brasileirao_a", odds, floor=0.0)
    mercados = {b.get("mercado") for b in bets}
    assert "Cartões" not in mercados                  # linha errada → pulado
    assert "Resultado" in mercados and "Escanteios" in mercados  # odds 10 → EV>0


def test_value_bets_matches_card_line_45():
    # se a casa oferecer 4.5, cartões DEVE entrar.
    odds = {"cards": {"line": 4.5, "Over": 10, "Under": 10}}
    bets = vb.value_bets_for_match("Flamengo", "Grêmio", "brasileirao_a", odds, floor=0.0)
    assert any(b.get("mercado") == "Cartões" for b in bets)


def test_value_bets_floor_respected():
    # odds próximas do "justo" (≈1/prob) → EV ~0 → não passa um piso alto.
    odds = {"1x2": {"H": 1.01, "D": 1.01, "A": 1.01}}
    bets = vb.value_bets_for_match("Flamengo", "Grêmio", "brasileirao_a", odds, floor=0.20)
    assert len(bets) == 0


# ─────────────────────────────── predictor ───────────────────────────────────
def test_predict_structure_and_lines():
    p = predictor.predict("Flamengo", "Grêmio", "brasileirao_a")
    assert "error" not in p
    pr = p["resultado"]["probs"]
    assert set(pr) == {"H", "D", "A"}
    assert abs(sum(pr.values()) - 1.0) < 0.02            # probabilidades somam ~1
    # LINHAS corretas (pega a regressão 6.5/10.5)
    assert set(p["cartoes"]["probs"]) == {"Under 4.5", "Over 4.5"}
    assert set(p["escanteios"]["probs"]) == {"Under 9.5", "Over 9.5"}


def test_predict_draw_decision_fields():
    p = predictor.predict("Flamengo", "Grêmio", "brasileirao_a")["resultado"]
    assert p["favorito"] in {"H", "D", "A"}
    assert isinstance(p["equilibrado"], bool)
    assert p["label"] in {"H", "D", "A"}


def test_predict_unknown_team_errors():
    p = predictor.predict("Time Que Nao Existe 123", "Grêmio", "brasileirao_a")
    assert "error" in p


def test_predict_cross_league_ficticio():
    p = predictor.predict("Flamengo", "Paris Saint-Germain",
                          home_league="brasileirao_a", away_league="ligue_1",
                          competition="Mundial")
    assert "error" not in p
    assert p["ficticio"] is True
    assert p["competition"] == "Mundial"
    assert p["home_league"] != p["away_league"]


def test_ensemble_loaded():
    assert len(predictor.get_predictor()._models) >= 1


# ──────────────────────────────── tools ──────────────────────────────────────
def test_tools_registered():
    for name in ["get_team_stats", "get_h2h", "get_league_table",
                 "get_team_schedule", "run_prediction_engine", "get_value_bets"]:
        assert name in tools.TOOL_MAP
    # schema deve ter as mesmas tools
    defined = {t["name"] for t in tools.TOOL_DEFINITIONS}
    assert defined == set(tools.TOOL_MAP)


def test_get_team_stats():
    r = tools.get_team_stats("Flamengo", league="brasileirao_a")
    assert "aproveitamento_pct" in r and "escanteios_avg" in r


def test_league_table_sorted_desc():
    t = tools.get_league_table("brasileirao_a")
    pts = [r["P"] for r in t if isinstance(r, dict) and "P" in r]
    assert len(pts) > 0 and pts == sorted(pts, reverse=True)


# ─────────────────────────────── runner ──────────────────────────────────────
def _run():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    ok = 0
    fail = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
            ok += 1
        except Exception as e:                          # noqa: BLE001
            fail += 1
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{'='*50}\n{ok} passaram, {fail} falharam, {len(tests)} no total")
    return 0 if fail == 0 else 1


if __name__ == "__main__":
    sys.exit(_run())
