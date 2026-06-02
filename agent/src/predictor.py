"""
predictor.py — carrega o modelo treinado e faz previsões.

Uso:
    from predictor import predict
    result = predict("Flamengo", "Grêmio", "brasileirao_a")
"""
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
import data_loader

_SRC_DIR   = Path(__file__).resolve().parent
MODELS_DIR = _SRC_DIR.parent / "models"
MODEL_PATH = MODELS_DIR / "oscabet_nn_v1.pt"
META_PATH  = MODELS_DIR / "oscabet_nn_v1_meta.json"


# ── Arquitetura idêntica ao notebook 03 ───────────────────────────────────────
class OscaBetNN(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.30),
            nn.Linear(256, 128),       nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.20),
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
        self._model   = None
        self._device  = "cuda" if torch.cuda.is_available() else "cpu"
        self._softmax = nn.Softmax(dim=1)
        self._load()

    def _load(self):
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Modelo não encontrado em {MODEL_PATH}\n"
                "Execute o notebook 03_treinamento.ipynb primeiro."
            )
        with open(META_PATH) as f:
            self._meta = json.load(f)

        ckpt = torch.load(MODEL_PATH, map_location=self._device, weights_only=False)
        self._model = OscaBetNN(input_dim=self._meta["input_dim"]).to(self._device)
        self._model.load_state_dict(ckpt["model_state_dict"])
        self._model.eval()
        print(f"✅ Modelo carregado ({self._meta['input_dim']} features, device={self._device})")

    def _get_team_row(self, team_name: str, side: str, league: str):
        """Retorna a linha de features mais recente do time no lado especificado."""
        feats = data_loader.features()
        mask  = (feats["league"] == league) & (feats[f"{side}_team"] == team_name)
        sub   = feats[mask].sort_values("date")
        return sub.iloc[-1] if len(sub) > 0 else None

    def _build_vector(self, home_row, away_row) -> list:
        feat_cols = self._meta["feat_cols"]
        x = []
        for col in feat_cols:
            if col.startswith("home_") and col in home_row.index:
                val = home_row[col]
            elif col.startswith("away_") and col in away_row.index:
                val = away_row[col]
            elif col.startswith("h2h_") and col in home_row.index:
                val = home_row[col]
            elif col.startswith("league_") and col in home_row.index:
                val = home_row[col]
            else:
                val = 0.5

            x.append(0.5 if (val is None or (isinstance(val, float) and np.isnan(val))) else float(val))
        return x

    def predict(self, home_team: str, away_team: str, league: str) -> dict:
        home_row = self._get_team_row(home_team, "home", league)
        away_row = self._get_team_row(away_team, "away", league)

        if home_row is None:
            return {"error": f"Time mandante '{home_team}' não encontrado na liga '{league}'."}
        if away_row is None:
            return {"error": f"Time visitante '{away_team}' não encontrado na liga '{league}'."}

        x = self._build_vector(home_row, away_row)
        t = torch.tensor([x], dtype=torch.float32).to(self._device)

        with torch.no_grad():
            lr, ly, lc = self._model(t)
            pr = self._softmax(lr).cpu().numpy()[0]
            py = self._softmax(ly).cpu().numpy()[0]
            pc = self._softmax(lc).cpu().numpy()[0]

        r_labels = ["H", "D", "A"]
        y_labels = ["Under 6.5",  "Over 6.5"]
        c_labels = ["Under 10.5", "Over 10.5"]


        return {
            "match":   f"{home_team} vs {away_team}",
            "league":  league,
            "resultado": {
                "probs":      {l: round(float(p), 3) for l, p in zip(r_labels, pr)},
                "label":      r_labels[int(pr.argmax())],
                "confidence": round(float(pr.max()), 3),
            },
            "cartoes": {
                "probs":      {l: round(float(p), 3) for l, p in zip(y_labels, py)},
                "label":      c_labels[int(py.argmax())],
                "confidence": round(float(py.max()), 3),
            },
            "escanteios": {
                "probs":      {l: round(float(p), 3) for l, p in zip(c_labels, pc)},
                "label":      c_labels[int(pc.argmax())],
                "confidence": round(float(pc.max()), 3),
            },
        }


# ── Singleton + função de atalho ──────────────────────────────────────────────
_instance: OscaBetPredictor | None = None

def get_predictor() -> OscaBetPredictor:
    global _instance
    if _instance is None:
        _instance = OscaBetPredictor()
    return _instance

def predict(home_team: str, away_team: str, league: str) -> dict:
    """Atalho direto — importa e chama sem instanciar manualmente."""
    return get_predictor().predict(home_team, away_team, league)
