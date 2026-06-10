"""
orchestrator.py — agentic loop do OscaBet (§3 do plano).

Fluxo:
    mensagem do usuário
        → LLM decide quais tools chamar
        → executor Python roda as tools
        → resultados injetados de volta na LLM
        → resposta final em linguagem natural (+ prediction se NN foi acionada)

Uso:
    from orchestrator import handle_chat
    result = handle_chat("Flamengo x Grêmio, quem vence?", history=[])
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import llm_client
from tools import TOOL_DEFINITIONS, execute_tool

# ── System Prompt (§6 do plano) ───────────────────────────────────────────────
SYSTEM_PROMPT = """\
Você é OscaBet, um especialista em futebol com acesso a dados históricos
detalhados de partidas, times e jogadores de múltiplas ligas.

╔═══════════════════════════════════════════════════════════════════════════╗
║ REGRA Nº 1 — ABSOLUTA E INEGOCIÁVEL                                        ║
║ Você NÃO possui conhecimento sobre quais times jogam em quais ligas,       ║
║ divisões ou temporadas. Qualquer "conhecimento" seu sobre isso está        ║
║ DESATUALIZADO e é PROIBIDO de usar. A ÚNICA fonte da verdade é a base de   ║
║ dados, acessível somente pelas tools.                                      ║
║                                                                            ║
║ Quando o usuário pedir QUALQUER previsão, palpite, SIMULAÇÃO, cenário      ║
║ hipotético, "e se", ou um confronto (mesmo FICTÍCIO) entre dois times, sua ║
║ ÚNICA ação permitida é CHAMAR a tool run_prediction_engine IMEDIATAMENTE,  ║
║ ANTES de escrever qualquer texto. NUNCA escreva uma análise/narrativa de   ║
║ um confronto sem antes chamar a tool para obter os números da rede neural. ║
║ É TERMINANTEMENTE PROIBIDO responder coisas como "o time não está na liga",║
║ "está em outra divisão" ou recusar a previsão SEM ANTES chamar a tool.     ║
║ Só depois que a tool retornar um erro é que você pode avisar o usuário.    ║
║ Recusar ou narrar sem chamar a tool é considerado um ERRO GRAVE.           ║
╚═══════════════════════════════════════════════════════════════════════════╝

Seu papel:
- Responder perguntas sobre futebol com precisão, usando os dados das tools.
- Dar opiniões concisas e embasadas, citando estatísticas relevantes.
- Quando houver previsão da rede neural, incorporá-la na resposta de forma clara
  (apresente como probabilidade, nunca como certeza absoluta).
- Ser direto e objetivo: máximo de 3-4 parágrafos por resposta.
- Nunca inventar estatísticas. Se não tiver dados, diga claramente.
- Responda SEMPRE em português do Brasil.

Ligas/competições disponíveis na base de dados:
  - brasileirao_a    → Brasileirão Série A
  - brasileirao_b    → Brasileirão Série B
  - copa_brasil      → Copa do Brasil
  - libertadores     → Copa Libertadores
  - premier_league   → Premier League (Inglaterra)
  - la_liga          → La Liga (Espanha)
  - serie_a          → Serie A (Itália)
  - bundesliga       → Bundesliga (Alemanha)
  - ligue_1          → Ligue 1 (França)
  - champions_league → UEFA Champions League

REGRAS IMPORTANTES sobre previsões:
- NUNCA recuse uma previsão dizendo que um time "não está na liga" ou "está em
  outra divisão". A base de dados é a fonte da verdade — SEMPRE chame a tool
  run_prediction_engine e deixe-a responder. Se o time realmente não existir, a
  tool retorna um erro e só então você avisa o usuário.
- Você PODE simular partidas FICTÍCIAS entre competições diferentes (ex.: Flamengo
  x PSG num "Mundial", ou um confronto de Libertadores). Para isso, chame a tool
  passando home_league e away_league (a liga de cada time) e, se quiser, competition
  como rótulo. Deixe claro na resposta que é um cenário hipotético.

Estilo: conversa natural, como um amigo que entende muito de futebol.
- Sem usar negrito, asteriscos ou marcadores.
- Sem dividir a resposta em seções com títulos como "Contexto" ou "Dados relevantes".
- Integre os números na narrativa de forma fluida, não como lista de métricas.
- Seja direto, use linguagem informal mas embasada.
"""

MAX_TOOL_ROUNDS = 5  # segurança contra loops infinitos


def handle_chat(message: str, history: list = None) -> dict:
    """
    Ponto de entrada principal do agente.

    Args:
        message : mensagem atual do usuário
        history : lista de turnos anteriores [{'role':..., 'content':...}]

    Returns:
        {
            'text':           str,         # resposta em linguagem natural
            'has_prediction': bool,        # True se a NN foi acionada
            'prediction':     dict | None, # dados brutos da previsão
            'tools_used':     list[str],   # nomes das tools chamadas
        }
    """
    if history is None:
        history = []

    # Monta contexto completo
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += history
    messages.append({"role": "user", "content": message})

    tools_used = []
    prediction = None
    response   = None

    # ── Agentic loop ─────────────────────────────────────────────────────────
    for round_num in range(MAX_TOOL_ROUNDS):

        response = llm_client.chat(messages=messages, tools=TOOL_DEFINITIONS)

        # Sem tool calls → LLM já tem tudo, sai do loop
        if not response["tool_calls"]:
            break

        # Registra o turno do assistente (com as tool calls)
        messages.append({
            "role":       "assistant",
            "content":    response["text"],
            "tool_calls": response["raw"].message.tool_calls,
        })

        # Executa cada tool e injeta o resultado
        for tc in response["tool_calls"]:
            name   = tc.function.name
            inputs = tc.function.arguments  # dict (ollama já deserializa)

            print(f"  🔧 [{round_num+1}] {name}({json.dumps(inputs, ensure_ascii=False)})")

            result = execute_tool(name, inputs)
            tools_used.append(name)

            # Captura prediction se a NN foi acionada com sucesso
            if name == "run_prediction_engine" and "error" not in result:
                prediction = result

            messages.append({
                "role":    "tool",
                "content": json.dumps(result, ensure_ascii=False, default=str),
            })

    # Se o último turno ainda tinha tool calls, chama sem tools para forçar resposta
    if response and response["tool_calls"]:
        response = llm_client.chat(messages=messages, tools=None)

    return {
        "text":           response["text"] if response else "",
        "has_prediction": prediction is not None,
        "prediction":     prediction,
        "tools_used":     list(set(tools_used)),
    }


# ── Modo terminal interativo (testes sem a API Flask) ─────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  OscaBet Agent — modo terminal")
    print("  Ctrl+C para sair")
    print("=" * 55)

    history = []

    while True:
        try:
            user_input = input("\nVocê: ").strip()
            if not user_input:
                continue

            print("\nOscaBet pensando...\n")
            result = handle_chat(user_input, history)

            print(f"OscaBet: {result['text']}")

            if result["has_prediction"]:
                p = result["prediction"]
                print(f"\n  ┌─ 📊 Previsão: {p.get('match')} ──────────────")
                print(f"  │  Resultado:  {p['resultado']['label']:8s} "
                      f"Over {p['resultado']['confidence']*100:.1f}%")
                print(f"  │  Cartões:    {p['cartoes']['label']:12s} "
                      f"({p['cartoes']['confidence']*100:.1f}%)")
                print(f"  │  Escanteios: {p['escanteios']['label']:12s} "
                      f"({p['escanteios']['confidence']*100:.1f}%)")
                print(f"  └{'─'*45}")

            if result["tools_used"]:
                print(f"\n  [tools: {', '.join(result['tools_used'])}]")

            # Mantém histórico para o próximo turno
            history.append({"role": "user",      "content": user_input})
            history.append({"role": "assistant",  "content": result["text"]})

        except KeyboardInterrupt:
            print("\n\nSaindo...")
            break
        except Exception as e:
            print(f"\n❌ Erro: {e}")
