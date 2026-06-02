"""
llm_client.py — wrapper do Ollama Cloud com suporte a Tool Use.

Configuração via .env na raiz do projeto:
    OLLAMA_API_KEY=sua_chave
    OLLAMA_HOST=https://ollama.com
    OLLAMA_MODEL=llama3.3:70b
"""
import os
from pathlib import Path
from ollama import Client
from dotenv import load_dotenv

# Carrega .env da raiz do projeto (src/ → agent/ → raiz)
_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(_ROOT / ".env")

HOST  = os.getenv("OLLAMA_HOST",  "https://ollama.com")
KEY   = os.getenv("OLLAMA_API_KEY", "")
MODEL = os.getenv("OLLAMA_MODEL", "llama3.3:70b")

if not KEY:
    raise EnvironmentError(
        "\n❌ OLLAMA_API_KEY não encontrada.\n"
        "Crie um arquivo .env na raiz do projeto com:\n\n"
        "  OLLAMA_API_KEY=sua_chave_aqui\n\n"
        "Gere sua key em: https://ollama.com/settings/keys"
    )

_client = Client(
    host=HOST,
    headers={"Authorization": f"Bearer {KEY}"},
)


def to_ollama_tools(tool_defs: list) -> list:
    """Converte schemas Claude API → formato Ollama/OpenAI."""
    return [
        {
            "type": "function",
            "function": {
                "name":        t["name"],
                "description": t["description"],
                "parameters":  t["input_schema"],
            },
        }
        for t in tool_defs
    ]


def chat(messages: list, tools: list = None) -> dict:
    """
    Envia mensagens para o Ollama Cloud.
    Retorna: { text, tool_calls, raw }
    """
    kwargs = {"model": MODEL, "messages": messages, "stream": False}
    if tools:
        kwargs["tools"] = to_ollama_tools(tools)

    response = _client.chat(**kwargs)
    msg      = response.message

    return {
        "text":       msg.content or "",
        "tool_calls": msg.tool_calls or [],
        "raw":        response,
    }


def ping() -> bool:
    """Testa a conexão. Retorna True se OK."""
    try:
        r = chat([{"role": "user", "content": "Responda apenas: ok"}])
        return len(r["text"]) > 0
    except Exception as e:
        print(f"❌ Falha: {e}")
        return False


if __name__ == "__main__":
    print(f"Host:  {HOST}\nModel: {MODEL}")
    print("Testando conexão...")
    print("✅ OK" if ping() else "❌ Falha")
