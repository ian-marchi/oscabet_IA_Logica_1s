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
