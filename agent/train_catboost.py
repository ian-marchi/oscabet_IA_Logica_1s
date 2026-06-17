# -*- coding: utf-8 -*-
"""
train_catboost.py — treina classificadores CatBoost (gradient boosting) para os
três mercados do OscaBet, como alternativa à rede neural.

Treina DUAS variantes e compara:
  - NUM : apenas as 103 features numéricas (comparação justa com a rede neural);
  - CAT : numéricas + categóricas nativas (liga, time mandante, time visitante).

Mesma divisão temporal (sem vazamento) e mesma ausência de pesos de classe da
receita da rede neural, para que a comparação seja honesta. Salva a variante CAT
em agent/models/catboost/ (result/cards/corners .cbm + meta).

Uso (no ambiente conda 'oscabet'):
    python agent/train_catboost.py
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import json
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.metrics import accuracy_score, log_loss

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
MODELS = BASE / "agent" / "models"
OUT = MODELS / "catboost"
OUT.mkdir(parents=True, exist_ok=True)

CUTOFF = "2024-03-01"
META_NN = json.load(open(MODELS / "oscabet_nn_v1_meta.json", encoding="utf-8"))
FEAT_COLS = META_NN["feat_cols"]            # mesmas 103 features da rede neural
CAT_COLS = ["league", "home_team", "away_team"]
TARGETS = {"result": 3, "yellow_cat": 2, "corners_cat": 2}
# baseline da rede neural (ensemble, backtest) p/ referência
NN_REF = {"result": 0.5181, "yellow_cat": 0.6215, "corners_cat": 0.5651}

feats = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
tg = pd.read_csv(PROC / "targets.csv", parse_dates=["match_date"])
feats[FEAT_COLS] = feats[FEAT_COLS].fillna(feats[FEAT_COLS].median())
for c in CAT_COLS:
    feats[c] = feats[c].astype(str).fillna("NA")

cut = pd.to_datetime(CUTOFF)
tr_mask = (feats["date"] < cut).values
va_mask = (feats["date"] >= cut).values


def split_xy(cols, target):
    y = tg[target].fillna(-1).astype(int).values
    Xtr = feats.loc[tr_mask, cols]; ytr = y[tr_mask]
    Xva = feats.loc[va_mask, cols]; yva = y[va_mask]
    mtr = ytr >= 0; mva = yva >= 0
    return Xtr[mtr], ytr[mtr], Xva[mva], yva[mva]


def train_one(target, n_classes, use_cat):
    cols = FEAT_COLS + (CAT_COLS if use_cat else [])
    Xtr, ytr, Xva, yva = split_xy(cols, target)
    cat_idx = [cols.index(c) for c in CAT_COLS] if use_cat else []
    loss = "MultiClass" if n_classes > 2 else "Logloss"
    model = CatBoostClassifier(
        iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3.0,
        loss_function=loss, eval_metric="Accuracy", random_seed=42,
        early_stopping_rounds=60, verbose=False, allow_writing_files=False,
    )
    model.fit(Pool(Xtr, ytr, cat_features=cat_idx), eval_set=Pool(Xva, yva, cat_features=cat_idx))
    proba = model.predict_proba(Xva)
    pred = proba.argmax(1)
    return model, accuracy_score(yva, pred), log_loss(yva, proba), cols, cat_idx


print(f"Treino: {tr_mask.sum():,} | Validação: {va_mask.sum():,} | Features num: {len(FEAT_COLS)}")
print("\nTreinando CatBoost (pode levar 1-3 min)…\n")

results = {}
saved = {}
for target, nc in TARGETS.items():
    m_num, acc_num, ll_num, _, _ = train_one(target, nc, use_cat=False)
    m_cat, acc_cat, ll_cat, cols, cat_idx = train_one(target, nc, use_cat=True)
    results[target] = (acc_num, ll_num, acc_cat, ll_cat)
    # salva a variante CAT (deployável)
    m_cat.save_model(str(OUT / f"{target}.cbm"))
    saved[target] = {"cols": cols, "cat_idx": cat_idx}
    print(f"  {target:12s} NN={NN_REF[target]:.4f} | CatBoost NUM={acc_num:.4f} | CatBoost CAT={acc_cat:.4f}")

# ── Tabela comparativa ────────────────────────────────────────────────────────
print("\n" + "=" * 74)
print(f"{'Mercado':<14}{'Rede Neural':>13}{'CatBoost NUM':>14}{'CatBoost CAT':>14}{'melhor':>12}")
print("-" * 74)
label = {"result": "Resultado", "yellow_cat": "Cartões", "corners_cat": "Escanteios"}
for target, (an, ln, ac, lc) in results.items():
    best = max([("NN", NN_REF[target]), ("NUM", an), ("CAT", ac)], key=lambda x: x[1])[0]
    print(f"{label[target]:<14}{NN_REF[target]:>13.4f}{an:>14.4f}{ac:>14.4f}{best:>12}")
print("=" * 74)

# ── Meta ──────────────────────────────────────────────────────────────────────
meta = {
    "backend": "catboost", "variant": "CAT (numéricas + liga/times categóricos)",
    "feat_cols": FEAT_COLS, "cat_cols": CAT_COLS,
    "all_cols": FEAT_COLS + CAT_COLS,
    "train_cutoff": CUTOFF, "targets": list(TARGETS),
    "metrics": {t: {"acuracia": round(results[t][2], 4), "log_loss": round(results[t][3], 4)} for t in TARGETS},
}
json.dump(meta, open(OUT / "meta.json", "w", encoding="utf-8"), indent=2, ensure_ascii=False)
print(f"\n✅ Modelos CatBoost (variante CAT) salvos em {OUT}")
