"""
predictor.py — carrega o modelo treinado (ENSEMBLE de seeds) e faz previsões.

Uso:
    from predictor import predict
    result = predict("Flamengo", "Grêmio", "brasileirao_a")

Notas:
- Suporta tanto o formato ENSEMBLE (lista de state_dicts, chave "ensemble")
  quanto o formato de modelo único (chave "model_state_dict"), retrocompatível.
- A previsão de RESULTADO usa uma camada de decisão para empates: como o empate
  raramente é o resultado isoladamente mais provável (~25% de base), o argmax puro
  quase nunca o escolhe. Quando o jogo é equilibrado (empate a poucos pontos do
  favorito), o "pick" passa a ser Empate e marcamos `equilibrado=True`. Isso é
  feito SOBRE as probabilidades (que seguem bem calibradas), sem custo de treino.
  Ajuste fino via env: DRAW_MARGIN (default 0.06) e EQUILIBRIO_MARGIN (0.08).
"""
import os
import json
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

import data_loader

_SRC_DIR   = Path(__file__).resolve().parent
MODELS_DIR = _SRC_DIR.parent / "models"
MODEL_PATH = MODELS_DIR / "oscabet_nn_v1.pt"
META_PATH  = MODELS_DIR / "oscabet_nn_v1_meta.json"

# Camada de decisão de empate (tunável por env)
DRAW_MARGIN       = float(os.getenv("DRAW_MARGIN", "0.05"))   # vira "Empate" se D a <=5pp do favorito
EQUILIBRIO_MARGIN = float(os.getenv("EQUILIBRIO_MARGIN", "0.08"))  # top1-top2 <=8pp => equilibrado


# ── Arquitetura idêntica ao notebook 03 ───────────────────────────────────────
class OscaBetNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.50),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.40),
            nn.Linear(128, 64),                             nn.ReLU(),
        )
        self.head_result  = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 3))
        self.head_yellow  = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))
        self.head_corners = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 2))

    def forward(self, x):
        h = self.backbone(x)
        return self.head_result(h), self.head_yellow(h), self.head_corners(h)


# ── Predictor principal ───────────────────────────────────────────────────────
class OscaBetPredictor:
    def __init__(self):
        self._meta    = None
        self._models  = []
        self._device  = "cuda" if torch.cuda.is_available() else "cpu"
        self._softmax = nn.Softmax(dim=1)
        self._load()

    def _load(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {MODEL_PATH}\n"
                "Execute agent/train_ensemble.py (ou o notebook 03) primeiro."
            )
        with open(META_PATH, encoding="utf-8") as f:
            self._meta = json.load(f)

        ckpt = torch.load(MODEL_PATH, map_location=self._device, weights_only=False)
        if "ensemble" in ckpt:
            states = ckpt["ensemble"]
        else:                                   # retrocompatível: modelo único
            states = [ckpt["model_state_dict"]]

        for st in states:
            m = OscaBetNN(input_dim=self._meta["input_dim"]).to(self._device)
            m.load_state_dict(st)
            m.eval()
            self._models.append(m)
        print(f"✅ Modelo carregado (ensemble de {len(self._models)} seed(s), "
              f"{self._meta['input_dim']} features, device={self._device})")

    def _avg_softmax(self, t):
        """Média das probabilidades softmax sobre todos os modelos do ensemble."""
        pr = py = pc = None
        with torch.no_grad():
            for m in self._models:
                lr, ly, lc = m(t)
                sr = self._softmax(lr).cpu().numpy()[0]
                sy = self._softmax(ly).cpu().numpy()[0]
                sc = self._softmax(lc).cpu().numpy()[0]
                pr = sr if pr is None else pr + sr
                py = sy if py is None else py + sy
                pc = sc if pc is None else pc + sc
        n = len(self._models)
        return pr / n, py / n, pc / n

    def _get_team_row(self, team_name: str, side: str, league: str = None):
        """Linha de features mais recente do time. Procura na liga indicada;
        se não achar (ou liga=None), procura em QUALQUER liga — permite partidas
        fictícias entre competições (ex.: Flamengo x PSG)."""
        feats = data_loader.features()
        base = feats[feats[f"{side}_team"] == team_name]
        if league:
            sub = base[base["league"] == league].sort_values("date")
            if len(sub) > 0:
                return sub.iloc[-1]
        sub = base.sort_values("date")
        return sub.iloc[-1] if len(sub) > 0 else None

    def _build_vector(self, home_row, away_row, neutralize_h2h: bool = False) -> list:
        feat_cols = self._meta["feat_cols"]
        x = []
        for col in feat_cols:
            if col.startswith("h2h_"):
                # Partida fictícia entre ligas: não há confronto direto → neutro.
                val = 0.5 if neutralize_h2h else (home_row[col] if col in home_row.index else 0.5)
            elif col.startswith("home_") and col in home_row.index:
                val = home_row[col]
            elif col.startswith("away_") and col in away_row.index:
                val = away_row[col]
            elif col.startswith("league_") and col in home_row.index:
                val = home_row[col]
            else:
                val = 0.5
            x.append(0.5 if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val))
        return x

    def predict(self, home_team: str, away_team: str, league: str = None,
                home_league: str = None, away_league: str = None,
                competition: str = None) -> dict:
        home_row = self._get_team_row(home_team, "home", home_league or league)
        away_row = self._get_team_row(away_team, "away", away_league or league)

        if home_row is None:
            return {"error": f"Time mandante '{home_team}' não encontrado na base de dados."}
        if away_row is None:
            return {"error": f"Time visitante '{away_team}' não encontrado na base de dados."}

        # Fictícia: times de ligas diferentes ou competição explícita (Mundial, etc.)
        ficticio = bool(competition) or (home_row["league"] != away_row["league"])
        x = self._build_vector(home_row, away_row, neutralize_h2h=ficticio)
        t = torch.tensor([x], dtype=torch.float32).to(self._device)
        pr, py, pc = self._avg_softmax(t)

        r_labels = ["H", "D", "A"]
        # Linhas REAIS usadas na construção dos alvos (notebook 02): 4.5 e 9.5.
        y_labels = ["Under 4.5", "Over 4.5"]
        c_labels = ["Under 9.5", "Over 9.5"]

        # ── Camada de decisão de empate ───────────────────────────────────────
        fav_idx   = int(pr.argmax())              # favorito estatístico
        p_draw    = float(pr[1])                  # índice 1 = Empate
        sorted_p  = np.sort(pr)[::-1]
        equilibrado = bool((sorted_p[0] - sorted_p[1]) <= EQUILIBRIO_MARGIN)

        if fav_idx != 1 and (pr.max() - p_draw) <= DRAW_MARGIN:
            pick_idx = 1                           # vira Empate em jogo apertado
        else:
            pick_idx = fav_idx

        return {
            "match":       f"{home_team} vs {away_team}",
            "league":      league or home_row["league"],
            "competition": competition,
            "ficticio":    ficticio,
            "home_league": home_row["league"],
            "away_league": away_row["league"],
            "resultado": {
                "probs":       {l: round(float(p), 3) for l, p in zip(r_labels, pr)},
                "label":       r_labels[pick_idx],         # pick (pode ser Empate)
                "favorito":    r_labels[fav_idx],          # favorito estatístico
                "equilibrado": equilibrado,
                "confidence":  round(float(pr[pick_idx]), 3),
            },
            "cartoes": {
                "probs":      {l: round(float(p), 3) for l, p in zip(y_labels, py)},
                "label":      y_labels[int(py.argmax())],
                "confidence": round(float(py.max()), 3),
            },
            "escanteios": {
                "probs":      {l: round(float(p), 3) for l, p in zip(c_labels, pc)},
                "label":      c_labels[int(pc.argmax())],
                "confidence": round(float(pc.max()), 3),
            },
        }


# ── Singleton + função de atalho ──────────────────────────────────────────────
_instance = None

def get_predictor() -> OscaBetPredictor:
    global _instance
    if _instance is None:
        _instance = OscaBetPredictor()
    return _instance

def predict(home_team: str, away_team: str, league: str = None,
            home_league: str = None, away_league: str = None,
            competition: str = None) -> dict:
    return get_predictor().predict(home_team, away_team, league,
                                   home_league, away_league, competition)
