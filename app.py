"""
app.py — API Flask do OscaBet + serve o frontend React.

Endpoints:
    GET  /              → frontend (web/index.html)
    GET  /api/health    → status do serviço e do modelo
    POST /api/chat      → {message, history} → resposta do agente

Como rodar (no ambiente conda 'oscabet', no Windows):
    conda activate oscabet
    python app.py
    # abre http://localhost:5000

O app apenas CONSOME o modelo via orchestrator/predictor — re-treinar a rede
(ex.: ensemble de seeds) só regenera o .pt e o app passa a usá-lo, sem mudanças
aqui.
"""
import os
# IMPORTANTE (Windows): definir antes de qualquer import que carregue torch/MKL.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import sys
import traceback
from pathlib import Path

# torch ANTES de pandas/numpy (evita OSError WinError 127 / shm.dll no Windows).
import torch  # noqa: F401

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from dotenv import load_dotenv

_ROOT = Path(__file__).resolve().parent
load_dotenv(_ROOT / ".env")

# Disponibiliza o pacote do agente para import
sys.path.insert(0, str(_ROOT / "agent" / "src"))
sys.path.insert(0, str(_ROOT / "agent"))          # para value_bets.py

WEB_DIR = _ROOT / "web"

app = Flask(__name__, static_folder=str(WEB_DIR), static_url_path="")
CORS(app)

# ── Carregamento preguiçoso do agente (para o /health funcionar mesmo sem chave) ─
_orchestrator = None
_load_error = None


def _get_orchestrator():
    """Importa o orchestrator sob demanda; cacheia erro de configuração."""
    global _orchestrator, _load_error
    if _orchestrator is not None or _load_error is not None:
        return _orchestrator
    try:
        import orchestrator  # importa llm_client (exige OLLAMA_API_KEY) + tools
        _orchestrator = orchestrator
    except Exception as e:  # noqa: BLE001
        _load_error = str(e)
        traceback.print_exc()
    return _orchestrator


# ── Rotas estáticas (frontend) ────────────────────────────────────────────────
@app.route("/")
def index():
    return send_from_directory(str(WEB_DIR), "index.html")


# ── API ───────────────────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    orch = _get_orchestrator()
    model_ok = False
    try:
        import predictor
        predictor.get_predictor()          # carrega o .pt (cacheado)
        model_ok = True
    except Exception:                       # noqa: BLE001
        model_ok = False
    return jsonify({
        "status":      "ok" if orch is not None else "degraded",
        "model_ready": model_ok,
        "agent_ready": orch is not None,
        "error":       _load_error,
        "model":       os.getenv("OLLAMA_MODEL", "?"),
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    orch = _get_orchestrator()
    if orch is None:
        return jsonify({
            "error": "Agente indisponível. Verifique OLLAMA_API_KEY no .env.",
            "detail": _load_error,
        }), 503

    data = request.get_json(silent=True) or {}
    message = (data.get("message") or "").strip()
    history = data.get("history") or []

    if not message:
        return jsonify({"error": "Mensagem vazia."}), 400

    try:
        result = orch.handle_chat(message, history)
        return jsonify(result)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": "Falha ao processar a mensagem.", "detail": str(e)}), 500


# ── Dashboard: times, previsão direta e rodada (jogos futuros) ────────────────
LEAGUE_LABELS = {
    "brasileirao_a": "Brasileirão A", "brasileirao_b": "Brasileirão B",
    "copa_brasil": "Copa do Brasil", "libertadores": "Libertadores",
    "premier_league": "Premier League", "la_liga": "La Liga", "serie_a": "Serie A",
    "bundesliga": "Bundesliga", "ligue_1": "Ligue 1", "champions_league": "Champions League",
}


@app.route("/api/leagues")
def leagues():
    return jsonify([{"key": k, "label": v} for k, v in LEAGUE_LABELS.items()])


@app.route("/api/teams")
def teams():
    """Lista os times de uma liga (rápido, vem do banco de features)."""
    league = request.args.get("league", "brasileirao_a")
    try:
        import data_loader
        f = data_loader.features()
        sub = f[f["league"] == league]
        names = sorted(set(sub["home_team"]) | set(sub["away_team"]))
        return jsonify({"league": league, "teams": names})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict")
def predict_endpoint():
    """Previsão DIRETA (sem LLM) — usada pelo simulador do dashboard."""
    home = (request.args.get("home") or "").strip()
    away = (request.args.get("away") or "").strip()
    if not home or not away:
        return jsonify({"error": "Informe home e away."}), 400
    try:
        import predictor
        pred = predictor.predict(
            home, away,
            league=request.args.get("league") or None,
            home_league=request.args.get("home_league") or None,
            away_league=request.args.get("away_league") or None,
            competition=request.args.get("competition") or None,
        )
        return jsonify(pred)
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/upcoming")
def upcoming():
    """Jogos futuros de uma liga + previsão + apostas de valor (pode demorar)."""
    league = request.args.get("league", "brasileirao_a")
    floor = float(request.args.get("floor", 0.05))
    max_m = int(request.args.get("max", 8))
    try:
        import value_bets as vb
        import predictor
        cli = vb.Odds()
        fixtures = cli.upcoming(league, max_m)
        out = []
        sem_odds = 0
        for fx in fixtures:
            pred = predictor.predict(fx["home"], fx["away"], league)
            if "error" in pred:
                continue
            raw = cli.get(f"{vb.BASE}/event/{fx['id']}/odds/1/all")
            odds = vb.parse_odds(raw)
            bets = vb.value_bets_for_match(fx["home"], fx["away"], league, odds, floor) if odds else []
            bets = [b for b in bets if "erro" not in b]
            if not odds:
                sem_odds += 1
            out.append({"id": fx["id"], "home": fx["home"], "away": fx["away"],
                        "ts": fx["ts"], "prediction": pred, "value_bets": bets,
                        "has_odds": bool(odds)})
        return jsonify({"league": league, "floor_pct": round(floor * 100, 1),
                        "jogos_sem_odds": sem_odds, "matches": out})
    except Exception as e:  # noqa: BLE001
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug = os.getenv("FLASK_ENV", "production") == "development"
    print("=" * 56)
    print("  OscaBet — servidor web")
    print(f"  http://localhost:{port}")
    print("=" * 56)
    # Warmup opcional do modelo (evita lag na 1ª previsão)
    try:
        import predictor
        predictor.get_predictor()
    except Exception:  # noqa: BLE001
        print("⚠ Modelo não pôde ser pré-carregado (veja /api/health).")
    app.run(host="0.0.0.0", port=port, debug=debug, use_reloader=False)
