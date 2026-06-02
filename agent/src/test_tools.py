"""
test_tools.py — testa as 5 tools sem precisar da LLM ou do orchestrator.

Execute dentro da pasta src/:
    python test_tools.py
"""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tools import (
    get_team_stats,
    get_h2h,
    get_league_table,
    get_team_schedule,
    run_prediction_engine,
)

def sep(title):
    print(f"\n{'═'*55}")
    print(f"  {title}")
    print('═'*55)

def pprint(obj):
    print(json.dumps(obj, indent=2, ensure_ascii=False, default=str))

# ── 1. Stats de um time ───────────────────────────────────────────────────────
sep("1. get_team_stats — Flamengo, últimos 10 jogos")
pprint(get_team_stats("Flamengo", last_n=10, league="brasileirao_a"))

# ── 2. H2H ────────────────────────────────────────────────────────────────────
sep("2. get_h2h — Flamengo × Grêmio")
pprint(get_h2h("Flamengo", "Grêmio"))

# ── 3. Tabela ─────────────────────────────────────────────────────────────────
sep("3. get_league_table — Brasileirão (top 5)")
table = get_league_table("brasileirao_a")
for row in table[:5]:
    print(f"  {row['pos']:2d}. {row['time']:<22s} "
          f"{row['P']:3d}pts  {row['V']}V {row['E']}E {row['D']}D  "
          f"SG{row['SG']:+d}  {row['aprov']}%")

# ── 4. Agenda ─────────────────────────────────────────────────────────────────
sep("4. get_team_schedule — Manchester City (últimos jogos)")
pprint(get_team_schedule("Manchester City", upcoming=False))

# ── 5. Previsão ───────────────────────────────────────────────────────────────
sep("5. run_prediction_engine — Flamengo × Grêmio")
pred = run_prediction_engine("Flamengo", "Grêmio", "brasileirao_a")
if "error" in pred:
    print(f"  ❌ {pred['error']}")
else:
    print(f"  Resultado  → {pred['resultado']['label']}  "
          f"({pred['resultado']['confidence']*100:.1f}%)")
    print(f"  Cartões    → {pred['cartoes']['label']}  "
          f"({pred['cartoes']['confidence']*100:.1f}%)")
    print(f"  Escanteios → {pred['escanteios']['label']}  "
          f"({pred['escanteios']['confidence']*100:.1f}%)")
    print()
    for mercado in ["resultado", "cartoes", "escanteios"]:
        probs = pred[mercado]["probs"]
        print(f"  {mercado.upper()}:")
        for label, prob in probs.items():
            bar = "█" * int(prob * 30)
            print(f"    {label:<8s} {prob*100:5.1f}%  {bar}")

sep("✅ Todos os testes concluídos")
