# -*- coding: utf-8 -*-
"""
evaluate_backtest.py — backtest temporal "como se fossem os próximos jogos".

Para CADA partida do período de validação (date >= cutoff), o ensemble prevê
usando apenas as features rolling DO PRÓPRIO JOGO (computadas a partir de jogos
ANTERIORES — sem vazamento). É exatamente o cenário de produção: prever a próxima
rodada com a informação disponível antes do apito inicial.

Gera (em evaluation/):
  - metrics.md                 tabela de métricas (3 mercados) + por temporada
  - confusion_matrices.png     matrizes de confusão
  - calibration.png            curvas de calibração
  - rolling_accuracy.png       acurácia móvel ao longo do tempo

Uso (no ambiente conda 'oscabet'):
    python agent/evaluate_backtest.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.calibration import calibration_curve

BASE   = Path(__file__).resolve().parent.parent
PROC   = BASE / "data" / "processed"
MODELS = BASE / "agent" / "models"
OUT    = BASE / "evaluation"
OUT.mkdir(exist_ok=True)

DRAW_MARGIN = float(os.getenv("DRAW_MARGIN", "0.05"))
YELL_LINE, CORN_LINE = 4.5, 9.5

meta = json.load(open(MODELS / "oscabet_nn_v1_meta.json", encoding="utf-8"))
COLS = meta["feat_cols"]
DIM  = meta["input_dim"]
CUTOFF = meta.get("train_cutoff", "2024-03-01")


class OscaBetNN(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.backbone = nn.Sequential(
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(128, 64), nn.ReLU())
        s.head_result  = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
        s.head_yellow  = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        s.head_corners = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(s, x):
        h = s.backbone(x); return s.head_result(h), s.head_yellow(h), s.head_corners(h)


ck = torch.load(MODELS / "oscabet_nn_v1.pt", map_location="cpu", weights_only=False)
states = ck.get("ensemble", [ck.get("model_state_dict")])
models = []
for st in states:
    m = OscaBetNN(DIM); m.load_state_dict(st); m.eval(); models.append(m)
print(f"Ensemble: {len(models)} modelo(s) | features: {DIM} | cutoff: {CUTOFF}")

feats = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
tg    = pd.read_csv(PROC / "targets.csv",  parse_dates=["match_date"])
va = (feats["date"] >= pd.to_datetime(CUTOFF)).values
X = feats[COLS].fillna(feats[COLS].median())

# alvos
yr = tg["result"].fillna(-1).astype(int).values
yy = tg["yellow_cat"].fillna(-1).astype(int).values
yc = tg["corners_cat"].fillna(-1).astype(int).values
valid = va & (yr >= 0) & (yy >= 0) & (yc >= 0)
Xv = X[valid].values
yr, yy, yc = yr[valid], yy[valid], yc[valid]
dates = feats.loc[valid, "date"].values
N = len(Xv)
print(f"Partidas de validação (backtest): {N:,}")

# ── Inferência do ensemble (média dos softmax) ───────────────────────────────
def ensemble(Xnp, bs=2048):
    PR = np.zeros((len(Xnp), 3)); PY = np.zeros((len(Xnp), 2)); PC = np.zeros((len(Xnp), 2))
    with torch.no_grad():
        for i in range(0, len(Xnp), bs):
            t = torch.tensor(Xnp[i:i+bs], dtype=torch.float32)
            for m in models:
                a, b, c = m(t)
                PR[i:i+bs] += torch.softmax(a, 1).numpy()
                PY[i:i+bs] += torch.softmax(b, 1).numpy()
                PC[i:i+bs] += torch.softmax(c, 1).numpy()
    n = len(models); return PR/n, PY/n, PC/n

PR, PY, PC = ensemble(Xv)

# picks
fav = PR.argmax(1)
draw_pick = fav.copy()
flip = (fav != 1) & ((PR.max(1) - PR[:, 1]) <= DRAW_MARGIN)
draw_pick[flip] = 1
predY = PY.argmax(1); predC = PC.argmax(1)

def maj_baseline(y):
    v, c = np.unique(y, return_counts=True); return c.max() / len(y)

acc_fav  = accuracy_score(yr, fav)
acc_draw = accuracy_score(yr, draw_pick)
acc_y    = accuracy_score(yy, predY)
acc_c    = accuracy_score(yc, predC)
ll_r = log_loss(yr, PR); ll_y = log_loss(yy, PY); ll_c = log_loss(yc, PC)

# ── metrics.md ────────────────────────────────────────────────────────────────
lines = []
lines.append(f"# Backtest temporal — OscaBet ({N:,} partidas reais)\n")
lines.append(f"Período: validação (`date >= {CUTOFF}`). Cada jogo previsto só com "
             f"features anteriores ao jogo (sem vazamento). Ensemble de {len(models)} seeds.\n")
lines.append("## Métricas por mercado\n")
lines.append("| Mercado | Acurácia | Baseline (classe maj.) | Log-loss |")
lines.append("|---|---|---|---|")
lines.append(f"| Resultado (favorito/argmax) | {acc_fav:.4f} | {maj_baseline(yr):.4f} | {ll_r:.4f} |")
lines.append(f"| Resultado (com regra de empate) | {acc_draw:.4f} | — | — |")
lines.append(f"| Cartões (O/U {YELL_LINE}) | {acc_y:.4f} | {maj_baseline(yy):.4f} | {ll_y:.4f} |")
lines.append(f"| Escanteios (O/U {CORN_LINE}) | {acc_c:.4f} | {maj_baseline(yc):.4f} | {ll_c:.4f} |")
lines.append("")
# por ano (representa o desvio de distribuição)
dfa = pd.DataFrame({"ano": pd.to_datetime(dates).year,
                    "r_ok": (fav == yr), "y_ok": (predY == yy), "c_ok": (predC == yc)})
g = dfa.groupby("ano").agg(jogos=("r_ok", "size"), resultado=("r_ok", "mean"),
                           cartoes=("y_ok", "mean"), escanteios=("c_ok", "mean"))
lines.append("## Acurácia por temporada (mostra a variação ano a ano)\n")
lines.append("| Ano | Jogos | Resultado | Cartões | Escanteios |")
lines.append("|---|---|---|---|---|")
for ano, row in g.iterrows():
    lines.append(f"| {ano} | {int(row.jogos)} | {row.resultado:.3f} | {row.cartoes:.3f} | {row.escanteios:.3f} |")
(OUT / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))

# ── Matrizes de confusão ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (name, yt, yp, labs) in zip(axes, [
    ("Resultado", yr, fav, ["Casa", "Empate", "Visitante"]),
    (f"Cartões O/U {YELL_LINE}", yy, predY, ["Under", "Over"]),
    (f"Escanteios O/U {CORN_LINE}", yc, predC, ["Under", "Over"]),
]):
    cm = confusion_matrix(yt, yp)
    cmp = cm / cm.sum(1, keepdims=True) * 100
    sns.heatmap(cmp, annot=[[f"{cm[i,j]}\n{cmp[i,j]:.0f}%" for j in range(len(labs))] for i in range(len(labs))],
                fmt="", cmap="Greens", xticklabels=labs, yticklabels=labs, ax=ax, cbar=False)
    ax.set_title(name); ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
plt.suptitle(f"Matrizes de confusão — backtest de {N:,} jogos reais", y=1.03)
plt.tight_layout(); plt.savefig(OUT / "confusion_matrices.png", dpi=130, bbox_inches="tight"); plt.close()

# ── Calibração ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (name, probs, truths, labs) in zip(axes, [
    ("Resultado", PR, yr, ["Casa", "Empate", "Visitante"]),
    (f"Cartões O/U {YELL_LINE}", PY, yy, ["Under", "Over"]),
    (f"Escanteios O/U {CORN_LINE}", PC, yc, ["Under", "Over"]),
]):
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfeita")
    for k, lab in enumerate(labs):
        pt, pp = calibration_curve((truths == k).astype(int), probs[:, k], n_bins=8, strategy="quantile")
        ax.plot(pp, pt, "o-", ms=4, label=lab)
    ax.set_title(name); ax.set_xlabel("Probabilidade prevista"); ax.set_ylabel("Frequência real")
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.suptitle("Calibração — quanto mais perto da diagonal, mais confiável a probabilidade", y=1.03)
plt.tight_layout(); plt.savefig(OUT / "calibration.png", dpi=130, bbox_inches="tight"); plt.close()

# ── Acurácia móvel ao longo do tempo ──────────────────────────────────────────
order = np.argsort(dates)
def rolling(ok, w=300):
    s = pd.Series(ok[order].astype(float)); return s.rolling(w, min_periods=w//2).mean().values
fig, ax = plt.subplots(figsize=(13, 4.5))
ax.plot(rolling(fav == yr), label="Resultado", color="#15E27F")
ax.plot(rolling(predY == yy), label=f"Cartões O/U {YELL_LINE}", color="#7C3AED")
ax.plot(rolling(predC == yc), label=f"Escanteios O/U {CORN_LINE}", color="#22D3EE")
for base, col in [(maj_baseline(yr), "#15E27F"), (maj_baseline(yy), "#7C3AED"), (maj_baseline(yc), "#22D3EE")]:
    ax.axhline(base, color=col, ls=":", lw=1, alpha=0.6)
ax.set_title("Acurácia móvel (janela de 300 jogos) ao longo do tempo — estabilidade do modelo")
ax.set_xlabel("Jogos (ordenados por data)"); ax.set_ylabel("Acurácia"); ax.legend(); ax.set_ylim(0.2, 0.9)
plt.tight_layout(); plt.savefig(OUT / "rolling_accuracy.png", dpi=130, bbox_inches="tight"); plt.close()

print(f"\n✅ Artefatos salvos em {OUT}/  (metrics.md, confusion_matrices.png, calibration.png, rolling_accuracy.png)")
