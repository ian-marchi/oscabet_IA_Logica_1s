# -*- coding: utf-8 -*-
"""
train_ensemble.py — treina um ENSEMBLE de K seeds da OscaBetNN e salva como
o modelo de produção (agent/models/oscabet_nn_v1.pt).

Por que ensemble: a média das probabilidades de várias seeds é mais estável e
mais bem calibrada que um único modelo. O peso de empate fica em 1.0 (sem viés
no treino — o tratamento de empate é feito na camada de decisão do predictor.py,
sem custo de acurácia).

Uso (no ambiente conda 'oscabet'):
    python agent/train_ensemble.py            # K=5 seeds padrão
    python agent/train_ensemble.py --seeds 99 7 123 2024 42 --w_draw 1.0
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix
from pathlib import Path
import argparse, json, warnings
warnings.filterwarnings("ignore")

ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[99, 7, 123, 2024, 42])
ap.add_argument("--w_draw", type=float, default=1.0)   # 1.0 = sem viés no treino
ap.add_argument("--cutoff", type=str, default="2024-03-01")
ap.add_argument("--nan_threshold", type=float, default=0.40)
args = ap.parse_args()

BASE = Path(__file__).resolve().parent.parent
PROC = BASE / "data" / "processed"
MODELS = BASE / "agent" / "models"
MODELS.mkdir(parents=True, exist_ok=True)

feats = pd.read_csv(PROC / "features.csv", parse_dates=["date"])
tg    = pd.read_csv(PROC / "targets.csv",  parse_dates=["match_date"])
META = ["match_id", "date", "league", "season", "home_team", "away_team"]
num = [c for c in feats.columns if c not in META and feats[c].dtype.kind in "fi"]
nan = feats[num].isnull().mean()
feat_cols = [c for c in num if nan[c] < args.nan_threshold]
X = feats[feat_cols].fillna(feats[feat_cols].median())

cut = pd.to_datetime(args.cutoff)
tr = (feats["date"] < cut).values
va = (feats["date"] >= cut).values
def gy(c, m): return tg.loc[m, c].fillna(-1).astype(int).values
Xtr, Xva = X[tr].values, X[va].values
yr_t, yr_v = gy("result", tr), gy("result", va)
yy_t, yy_v = gy("yellow_cat", tr), gy("yellow_cat", va)
yc_t, yc_v = gy("corners_cat", tr), gy("corners_cat", va)
vt = (yr_t >= 0) & (yy_t >= 0) & (yc_t >= 0)
vv = (yr_v >= 0) & (yy_v >= 0) & (yc_v >= 0)
Xtr, yr_t, yy_t, yc_t = Xtr[vt], yr_t[vt], yy_t[vt], yc_t[vt]
Xva, yr_v, yy_v, yc_v = Xva[vv], yr_v[vv], yy_v[vv], yc_v[vv]
print(f"Treino: {len(Xtr):,} | Val: {len(Xva):,} | Features: {Xtr.shape[1]}")


class DS(Dataset):
    def __init__(s, X, a, b, c):
        s.X = torch.tensor(X, dtype=torch.float32); s.a = torch.tensor(a); s.b = torch.tensor(b); s.c = torch.tensor(c)
    def __len__(s): return len(s.X)
    def __getitem__(s, i): return s.X[i], s.a[i], s.b[i], s.c[i]


class OscaBetNN(nn.Module):
    def __init__(s, d):
        super().__init__()
        s.backbone = nn.Sequential(
            nn.Linear(d, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.50),
            nn.Linear(256, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.40),
            nn.Linear(128, 64), nn.ReLU())
        s.head_result  = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
        s.head_yellow  = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        s.head_corners = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
    def forward(s, x):
        h = s.backbone(x); return s.head_result(h), s.head_yellow(h), s.head_corners(h)


def train_one(seed):
    torch.manual_seed(seed); np.random.seed(seed)
    trl = DataLoader(DS(Xtr, yr_t, yy_t, yc_t), batch_size=256, shuffle=True, drop_last=True)
    val = DataLoader(DS(Xva, yr_v, yy_v, yc_v), batch_size=512, shuffle=False)
    m = OscaBetNN(Xtr.shape[1])
    wr = torch.tensor([1.0, args.w_draw, 1.0], dtype=torch.float32)
    cr = nn.CrossEntropyLoss(weight=wr); cy = nn.CrossEntropyLoss(); cc = nn.CrossEntropyLoss()
    opt = torch.optim.Adam(m.parameters(), lr=6e-4, weight_decay=8e-4)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=8, min_lr=1e-5)
    best, best_state, wait = -1, None, 0
    for ep in range(1, 251):
        m.train()
        for xb, a, b, c in trl:
            opt.zero_grad(); lr_, ly_, lc_ = m(xb)
            (cr(lr_, a) + cy(ly_, b) + cc(lc_, c)).backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        m.eval(); P = [[], [], []]; T = [[], [], []]
        with torch.no_grad():
            for xb, a, b, c in val:
                o = m(xb)
                for k in range(3): P[k].append(o[k].argmax(1).numpy())
                T[0].append(a.numpy()); T[1].append(b.numpy()); T[2].append(c.numpy())
        accs = [accuracy_score(np.concatenate(T[k]), np.concatenate(P[k])) for k in range(3)]
        mean = sum(accs) / 3; sch.step(mean)
        if mean > best: best, best_state, wait = mean, {k: v.cpu().clone() for k, v in m.state_dict().items()}, 0
        else: wait += 1
        if wait >= 45: break
    m.load_state_dict(best_state); m.eval()
    return m, best_state


# ── Treina o ensemble ─────────────────────────────────────────────────────────
models, states = [], []
for s in args.seeds:
    print(f"  treinando seed {s}…", flush=True)
    m, st = train_one(s); models.append(m); states.append(st)


def ensemble_softmax(Xnp):
    t = torch.tensor(Xnp, dtype=torch.float32)
    pr = np.zeros((len(Xnp), 3)); py = np.zeros((len(Xnp), 2)); pc = np.zeros((len(Xnp), 2))
    with torch.no_grad():
        for m in models:
            a, b, c = m(t)
            pr += torch.softmax(a, 1).numpy(); py += torch.softmax(b, 1).numpy(); pc += torch.softmax(c, 1).numpy()
    n = len(models); return pr / n, py / n, pc / n


pr, py, pc = ensemble_softmax(Xva)
acc_r = accuracy_score(yr_v, pr.argmax(1)); acc_y = accuracy_score(yy_v, py.argmax(1)); acc_c = accuracy_score(yc_v, pc.argmax(1))
ll_r = log_loss(yr_v, pr); ll_y = log_loss(yy_v, py); ll_c = log_loss(yc_v, pc)
print("\n── Ensemble (média de %d seeds) — validação ──" % len(models))
print(f"  resultado  acc={acc_r:.4f}  logloss={ll_r:.4f}")
print(f"  cartões    acc={acc_y:.4f}  logloss={ll_y:.4f}")
print(f"  escanteios acc={acc_c:.4f}  logloss={ll_c:.4f}")
print("  confusão resultado (linhas=real H/D/A):")
print(confusion_matrix(yr_v, pr.argmax(1), labels=[0, 1, 2]))

# ── Salva o ensemble ──────────────────────────────────────────────────────────
mp = MODELS / "oscabet_nn_v1.pt"
jp = MODELS / "oscabet_nn_v1_meta.json"
metrics = {
    "resultado":  {"acuracia": float(acc_r), "log_loss": float(ll_r)},
    "cartoes":    {"acuracia": float(acc_y), "log_loss": float(ll_y)},
    "escanteios": {"acuracia": float(acc_c), "log_loss": float(ll_c)},
}
torch.save({
    "ensemble":   states,                 # lista de state_dicts (K seeds)
    "n_models":   len(states),
    "seeds":      args.seeds,
    "input_dim":  Xtr.shape[1],
    "feat_cols":  feat_cols,
    "train_cutoff": args.cutoff,
    "w_draw":     args.w_draw,
    "metrics":    metrics,
}, mp)
with open(jp, "w", encoding="utf-8") as f:
    json.dump({
        "input_dim": int(Xtr.shape[1]), "feat_cols": feat_cols, "train_cutoff": args.cutoff,
        "n_models": len(states), "seeds": args.seeds, "w_draw": args.w_draw,
        "n_train": int(len(Xtr)), "n_val": int(len(Xva)),
        "metrics": {k: {m: round(float(v), 4) for m, v in val.items()} for k, val in metrics.items()},
    }, f, indent=2, ensure_ascii=False)
print(f"\n✅ Ensemble salvo ({len(states)} modelos) em {mp}")
