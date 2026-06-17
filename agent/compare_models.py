# -*- coding: utf-8 -*-
"""
compare_models.py — comparação JUSTA Rede Neural × CatBoost no MESMO conjunto de
validação (backtest temporal, sem vazamento). Gera evaluation/model_comparison.md.

Uso: python agent/compare_models.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
from pathlib import Path

import torch                      # torch ANTES de pandas/numpy (Windows: WinError 127)
import torch.nn as nn
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, log_loss

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
MODELS = BASE / "agent" / "models"
OUT = BASE / "evaluation"; OUT.mkdir(exist_ok=True)
CUTOFF = "2024-03-01"

meta = json.load(open(MODELS / "oscabet_nn_v1_meta.json", encoding="utf-8"))
FEAT = meta["feat_cols"]; DIM = meta["input_dim"]
CAT = ["league", "home_team", "away_team"]
TARGETS = {"result": "Resultado", "yellow_cat": "Cartões (O/U 4,5)", "corners_cat": "Escanteios (O/U 9,5)"}

feats = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
tg = pd.read_csv(PROC / "targets.csv", parse_dates=["match_date"])
feats[FEAT] = feats[FEAT].fillna(feats[FEAT].median())
for c in CAT:
    feats[c] = feats[c].astype(str)
va = (feats["date"] >= pd.to_datetime(CUTOFF)).values


class Net(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.backbone = nn.Sequential(nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.4), nn.Linear(128, 64), nn.ReLU())
        s.head_result = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
        s.head_yellow = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        s.head_corners = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(s, x): h = s.backbone(x); return s.head_result(h), s.head_yellow(h), s.head_corners(h)


# Rede neural (ensemble)
ck = torch.load(MODELS / "oscabet_nn_v1.pt", map_location="cpu", weights_only=False)
states = ck.get("ensemble", [ck.get("model_state_dict")])
nets = [Net(DIM) for _ in states]
for m, st in zip(nets, states):
    m.load_state_dict(st); m.eval()

# CatBoost
cb = {t: CatBoostClassifier() for t in TARGETS}
for t in TARGETS:
    cb[t].load_model(str(MODELS / "catboost" / f"{t}.cbm"))

Xnum = feats.loc[va, FEAT].values
Xcat = feats.loc[va, FEAT + CAT]


def nn_probs(Xnp, bs=4096):
    P = [np.zeros((len(Xnp), n)) for n in (3, 2, 2)]
    with torch.no_grad():
        for i in range(0, len(Xnp), bs):
            tt = torch.tensor(Xnp[i:i+bs], dtype=torch.float32)
            for m in nets:
                o = m(tt)
                for k in range(3): P[k][i:i+bs] += torch.softmax(o[k], 1).numpy()
    return [p / len(nets) for p in P]


PRnn, PYnn, PCnn = nn_probs(Xnum)
probs_nn = {"result": PRnn, "yellow_cat": PYnn, "corners_cat": PCnn}
probs_cb = {t: cb[t].predict_proba(Xcat) for t in TARGETS}

lines = ["# Comparação Rede Neural × CatBoost (backtest temporal)\n"]
n_val = int(va.sum())
lines.append(f"Conjunto de validação: {n_val:,} partidas (`date >= {CUTOFF}`), sem vazamento.\n")
lines.append("| Mercado | Métrica | Rede Neural | CatBoost | Δ |")
lines.append("|---|---|---|---|---|")
print(f"{'Mercado':<22}{'NN acc':>9}{'CB acc':>9}{'NN ll':>9}{'CB ll':>9}")
print("-" * 58)
for t, lbl in TARGETS.items():
    y = tg.loc[va, t].fillna(-1).astype(int).values
    m = y >= 0
    pn, pc = probs_nn[t][m], probs_cb[t][m]
    yy = y[m]
    an, ac = accuracy_score(yy, pn.argmax(1)), accuracy_score(yy, pc.argmax(1))
    ln, lc = log_loss(yy, pn), log_loss(yy, pc)
    print(f"{lbl:<22}{an:>9.4f}{ac:>9.4f}{ln:>9.4f}{lc:>9.4f}")
    lines.append(f"| {lbl} | Acurácia | {an:.4f} | {ac:.4f} | {ac-an:+.4f} |")
    lines.append(f"| {lbl} | Log-loss | {ln:.4f} | {lc:.4f} | {lc-ln:+.4f} |")
(OUT / "model_comparison.md").write_text("\n".join(lines), encoding="utf-8")
print(f"\n✅ Salvo: {OUT/'model_comparison.md'}")
