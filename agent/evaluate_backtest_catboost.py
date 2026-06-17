# -*- coding: utf-8 -*-
"""
evaluate_backtest_catboost.py — versão CatBoost do backtest temporal, para gerar as
figuras do artigo a partir do modelo PADRÃO (CatBoost), e não da rede neural.

Replica EXATAMENTE o pré-processamento do train_catboost.py (fill numérico pela
mediana global; categóricas como str) e a mesma divisão temporal (sem vazamento).

Gera (em evaluation/), sobrescrevendo as versões da rede neural:
  - confusion_matrices.png   matrizes de confusão (CatBoost)
  - calibration.png          curvas de calibração (CatBoost)

Uso (no ambiente conda 'oscabet'):
    python agent/evaluate_backtest_catboost.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from sklearn.calibration import calibration_curve

BASE   = Path(__file__).resolve().parent.parent
PROC   = BASE / "data" / "processed"
CBDIR  = BASE / "agent" / "models" / "catboost"
OUT    = BASE / "evaluation"
OUT.mkdir(exist_ok=True)

YELL_LINE, CORN_LINE = 4.5, 9.5
meta = json.load(open(CBDIR / "meta.json", encoding="utf-8"))
FEAT_COLS = meta["feat_cols"]
CAT_COLS  = meta["cat_cols"]
ALL_COLS  = meta["all_cols"]
CUTOFF    = meta.get("train_cutoff", "2024-03-01")

# ── Modelos CatBoost (1 por mercado) ──────────────────────────────────────────
TARGETS = {"result": ("Resultado", ["Casa", "Empate", "Visitante"]),
           "yellow_cat": (f"Cartões O/U {YELL_LINE}", ["Under", "Over"]),
           "corners_cat": (f"Escanteios O/U {CORN_LINE}", ["Under", "Over"])}
cb = {}
for t in TARGETS:
    m = CatBoostClassifier(); m.load_model(str(CBDIR / f"{t}.cbm")); cb[t] = m
print(f"CatBoost: {len(cb)} modelos | features: {len(FEAT_COLS)} num + {len(CAT_COLS)} cat | cutoff: {CUTOFF}")

# ── Pré-processamento IDÊNTICO ao treino ──────────────────────────────────────
feats = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
tg    = pd.read_csv(PROC / "targets.csv",  parse_dates=["match_date"])
feats[FEAT_COLS] = feats[FEAT_COLS].fillna(feats[FEAT_COLS].median())
for c in CAT_COLS:
    feats[c] = feats[c].astype(str).fillna("NA")

va = (feats["date"] >= pd.to_datetime(CUTOFF)).values
yr = tg["result"].fillna(-1).astype(int).values
yy = tg["yellow_cat"].fillna(-1).astype(int).values
yc = tg["corners_cat"].fillna(-1).astype(int).values
valid = va & (yr >= 0) & (yy >= 0) & (yc >= 0)
Xv = feats.loc[valid, ALL_COLS]
yr, yy, yc = yr[valid], yy[valid], yc[valid]
N = len(Xv)
print(f"Partidas de validação (backtest): {N:,}")


def proba_aligned(model, X, n_classes):
    """predict_proba com colunas alinhadas ao índice da classe (0..n-1)."""
    raw = model.predict_proba(X)
    classes = list(model.classes_)
    out = np.zeros((len(X), n_classes))
    for col, cls in enumerate(classes):
        out[:, int(cls)] = raw[:, col]
    return out


PR = proba_aligned(cb["result"], Xv, 3)
PY = proba_aligned(cb["yellow_cat"], Xv, 2)
PC = proba_aligned(cb["corners_cat"], Xv, 2)
predR, predY, predC = PR.argmax(1), PY.argmax(1), PC.argmax(1)
print(f"Acurácias — resultado {accuracy_score(yr, predR):.4f} | "
      f"cartões {accuracy_score(yy, predY):.4f} | escanteios {accuracy_score(yc, predC):.4f}")
print(f"Log-loss  — resultado {log_loss(yr, PR):.4f} | "
      f"cartões {log_loss(yy, PY):.4f} | escanteios {log_loss(yc, PC):.4f}")

# ── Matrizes de confusão ──────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (yt, yp, (name, labs)) in zip(axes, [
    (yr, predR, TARGETS["result"]),
    (yy, predY, TARGETS["yellow_cat"]),
    (yc, predC, TARGETS["corners_cat"]),
]):
    cm = confusion_matrix(yt, yp)
    cmp = cm / cm.sum(1, keepdims=True) * 100
    sns.heatmap(cmp, annot=[[f"{cm[i,j]}\n{cmp[i,j]:.0f}%" for j in range(len(labs))] for i in range(len(labs))],
                fmt="", cmap="Greens", xticklabels=labs, yticklabels=labs, ax=ax, cbar=False)
    ax.set_title(name); ax.set_xlabel("Previsto"); ax.set_ylabel("Real")
plt.suptitle(f"Matrizes de confusão — CatBoost, backtest de {N:,} jogos reais", y=1.03)
plt.tight_layout(); plt.savefig(OUT / "confusion_matrices.png", dpi=130, bbox_inches="tight"); plt.close()

# ── Calibração ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
for ax, (probs, truths, (name, labs)) in zip(axes, [
    (PR, yr, TARGETS["result"]),
    (PY, yy, TARGETS["yellow_cat"]),
    (PC, yc, TARGETS["corners_cat"]),
]):
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfeita")
    for k, lab in enumerate(labs):
        pt, pp = calibration_curve((truths == k).astype(int), probs[:, k], n_bins=8, strategy="quantile")
        ax.plot(pp, pt, "o-", ms=4, label=lab)
    ax.set_title(name); ax.set_xlabel("Probabilidade prevista"); ax.set_ylabel("Frequência real")
    ax.legend(fontsize=8); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
plt.suptitle("Calibração (CatBoost) — quanto mais perto da diagonal, mais confiável a probabilidade", y=1.03)
plt.tight_layout(); plt.savefig(OUT / "calibration.png", dpi=130, bbox_inches="tight"); plt.close()

print(f"\n✅ Figuras CatBoost salvas em {OUT}/  (confusion_matrices.png, calibration.png)")
