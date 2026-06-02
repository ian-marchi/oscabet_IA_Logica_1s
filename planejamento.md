# OscaBet Agent — Planejamento do Projeto

> **Trabalho de IA Agêntica — 1º Período**
> Especialista de futebol com chat em linguagem natural, previsões por rede neural e auto-atualização de banco de dados.

---

## 1. Visão Geral do Projeto

### Problema resolvido
Um agente inteligente de futebol com duas capacidades integradas:

1. **Especialista conversacional** — Responde qualquer pergunta sobre futebol (times, jogadores, ligas, torneios, desempenho, história) usando uma LLM alimentada pelo banco de dados histórico coletado.

2. **Motor de previsão** — Quando a pergunta envolve um jogo futuro, aciona a rede neural treinada para prever resultado, cartões amarelos e escanteios, retornando a previsão junto com a resposta.

### Público-alvo
Apostadores esportivos, jornalistas esportivos e entusiastas de futebol que querem um especialista analítico disponível via chat.

### Objetivo do agente
Receber qualquer mensagem em linguagem natural sobre futebol, identificar automaticamente se é uma pergunta geral ou uma solicitação de previsão, e responder com informações precisas, opiniões embasadas e/ou previsões quantitativas com justificativa.

### Padrões de IA Agêntica utilizados
| Padrão | Onde aparece |
|---|---|
| **Routing / Roteamento** | Botão manual no chat aciona o `orchestrator.py`, que roteia para a NN e injeta o resultado de volta na LLM |
| **Prompt Chaining** | LLM → classificação → (NN opcionalmente) → LLM formata resposta final |
| **Tool Use** | LLM usa ferramentas Python para buscar stats, calcular features e chamar a NN |
| **Sequential Workflow** | Pipeline: mensagem → intent → dados → inferência → resposta |
| **Human-in-the-Loop** | Chat permite que o usuário refine a pergunta; previsões podem ser refeitas |

---

## 2. Arquitetura do Monorepo

```
oscabet-agent/
│
├── README.md
├── PLANEJAMENTO.md
├── .gitignore
├── .env.example
│
├── scraper/                            ← Coleta de dados do Sofascore
│   ├── environment.yml
│   ├── config.py
│   ├── run_scraper.py                  ← Coleta completa
│   ├── resume_scraper.py               ← Retoma de onde parou
│   └── src/
│       ├── client.py                   ← Session HTTP + rate limit + headers
│       ├── endpoints.py                ← Todas as URLs do Sofascore centralizadas
│       ├── collector.py                ← Orquestra ligas e temporadas
│       ├── match_scraper.py            ← Coleta partida por partida
│       ├── storage.py                  ← Salva/atualiza CSVs incrementalmente
│       └── logger.py                   ← Log de progresso e erros
│
├── data/
│   ├── raw/                            ← CSVs brutos do scraper
│   │   ├── matches.csv
│   │   ├── match_stats.csv
│   │   └── scraper_log.json
│   ├── processed/                      ← Features calculadas para treino
│   │   ├── features.csv
│   │   └── targets.csv
│   └── backups/                        ← [NOVO] Snapshots automáticos do banco
│       └── YYYY-MM-DD_HH-MM/
│           ├── matches.csv
│           └── match_stats.csv
│
├── agent/                              ← Núcleo da IA
│   ├── environment.yml
│   ├── src/
│   │   ├── orchestrator.py             ← [NOVO] Cérebro central: roteia LLM ↔ NN (acionado pelo botão)
│   │   ├── llm_client.py               ← [NOVO] Wrapper da LLM (Claude API)
│   │   ├── tools.py                    ← [NOVO] Ferramentas que a LLM pode chamar
│   │   ├── data_loader.py
│   │   ├── preprocessor.py
│   │   ├── model.py                    ← Rede neural (3 cabeças: resultado/cartões/escanteios)
│   │   ├── trainer.py
│   │   ├── predictor.py
│   │   ├── explainer.py
│   │   └── router.py
│   ├── notebooks/
│   │   ├── 01_exploracao_dados.ipynb
│   │   ├── 02_preprocessamento.ipynb
│   │   ├── 03_treinamento.ipynb
│   │   └── 04_testes_agente.ipynb
│   ├── models/
│   │   └── oscabet_nn_v1.pt
│   └── tests/
│       ├── test_orchestrator.py
│       ├── test_predictor.py
│       └── test_explainer.py
│
├── updater/                            ← [NOVO] Sistema de auto-atualização
│   ├── update_manager.py               ← Orquestra: backup → scraping → swap
│   ├── backup.py                       ← Cria snapshot timestampado do banco
│   ├── db_swap.py                      ← Substitui banco ativo pelo novo
│   └── scheduler.py                    ← Agenda atualizações automáticas (cron)
│
├── api/                                ← Backend Flask
│   ├── requirements.txt
│   ├── app.py
│   ├── config.py
│   └── routes/
│       ├── __init__.py
│       ├── chat.py                     ← [NOVO] POST /api/chat (principal)
│       ├── predict.py                  ← POST /api/predict (direto, sem chat)
│       ├── update.py                   ← [NOVO] POST /api/update (dispara updater)
│       ├── history.py                  ← GET /api/history/:team
│       └── health.py                   ← GET /api/health
│
└── web/                                ← Frontend React
    ├── package.json
    ├── vite.config.js
    └── src/
        ├── main.jsx
        ├── App.jsx
        ├── pages/
        │   ├── Chat.jsx                ← [NOVO] Página principal: chat com o agente
        │   └── Admin.jsx               ← [NOVO] Painel de atualização do banco
        └── components/
            ├── ChatWindow.jsx          ← Interface de chat (mensagens + input)
            ├── MessageBubble.jsx       ← Renderiza texto e PredictionCard quando presente
            ├── PredictionCard.jsx      ← Card dos 3 mercados (aparece dentro do chat)
            ├── ExplanationPanel.jsx    ← Justificativa expansível
            ├── UpdatePanel.jsx         ← Status e acionamento do updater
            └── ConfirmationModal.jsx
```

---

## 3. Arquitetura do Agente — Duas Engines Integradas

O usuário interage exclusivamente via **chat em linguagem natural**. A LLM decide sozinha, a cada mensagem, quais ferramentas chamar — incluindo quando acionar a rede neural. Isso é o padrão **Tool Use** da IA agêntica aplicado de forma completa.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         USUÁRIO (Chat Web)                          │
│                                                                     │
│  [Campo de texto livre — qualquer mensagem sobre futebol]           │
│  "Flamengo x Grêmio no domingo, quem você acha que vence?"          │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ POST /api/chat
                               │ { message, history }
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        orchestrator.py                              │
│                                                                     │
│  Passa a mensagem + histórico para a LLM junto com a lista de       │
│  TODAS as tools disponíveis (definições de schema)                  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     LLM (Claude API)                                │
│                                                                     │
│  Analisa a mensagem e decide quais tools chamar:                    │
│                                                                     │
│  Pergunta geral → chama get_team_stats(), get_h2h(), etc.           │
│  Pergunta sobre jogo futuro → chama run_prediction_engine()         │
│                               + get_h2h() + get_team_stats()        │
│                                                                     │
│  A LLM retorna chamadas de tool no formato:                         │
│  [{ "tool": "run_prediction_engine",                                │
│     "input": { "home": "Flamengo", "away": "Grêmio",               │
│                "league": "brasileirao_a" } }]                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ tool_calls detectados
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      orchestrator.py                                │
│                   (executor das tool calls)                         │
│                                                                     │
│  Para cada tool call retornada pela LLM:                            │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ run_prediction_engine()  →  router → data_loader             │   │
│  │                              → preprocessor → predictor      │   │
│  │                              → explainer                     │   │
│  │                              → retorna dict com probs        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │ get_team_stats()    →  consulta features.csv                 │   │
│  │ get_h2h()           →  filtra confrontos diretos             │   │
│  │ get_player_stats()  →  agrega stats individuais              │   │
│  │ get_league_table()  →  lê tabela da liga                     │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                     │
│  Resultados de todas as tools são injetados de volta na LLM         │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ tool results injetados
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                  LLM — Geração da Resposta Final                    │
│                                                                     │
│  Com todos os dados em mãos, gera resposta em linguagem natural.    │
│  Quando run_prediction_engine foi chamada, inclui no retorno        │
│  um bloco JSON estruturado sinalizando ao frontend que deve         │
│  renderizar o PredictionCard.                                       │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
                      Usuário recebe resposta
              (texto puro OU texto + PredictionCard)
```

### 3.1 Como a LLM reconhece que deve chamar a NN

A LLM recebe no system prompt a definição de cada tool com exemplos de quando usá-las. O schema da `run_prediction_engine` inclui:

```python
{
  "name": "run_prediction_engine",
  "description": (
    "Aciona a rede neural treinada para prever o resultado, total de cartões "
    "amarelos e total de escanteios de uma partida de futebol ainda não realizada. "
    "Use esta tool SEMPRE que o usuário mencionar dois times e um contexto de "
    "jogo futuro — mesmo que implícito, como 'quem você acha que ganha', "
    "'vai ter muitos escanteios', 'apostaria em quê'. "
    "NÃO use para jogos já realizados."
  ),
  "input_schema": {
    "home_team": "Nome do time mandante",
    "away_team": "Nome do time visitante",
    "league":    "Liga/campeonato (chave do config, ex: 'brasileirao_a')"
  }
}
```

A LLM usa esse description para decidir sozinha. Não há nenhuma lógica de detecção de intenção no código Python — essa responsabilidade fica inteiramente com a inteligência da LLM.

### 3.2 Fluxo de mensagens no agentic loop

```python
# orchestrator.py — loop de execução das tools

def handle_chat(message: str, history: list) -> dict:
    tools = load_tool_definitions()        # schemas de todas as tools
    messages = history + [{"role": "user", "content": message}]

    # 1ª chamada: LLM decide quais tools usar
    response = llm_client.call(messages=messages, tools=tools)

    # Loop: executa as tools que a LLM pediu
    while response.has_tool_calls:
        tool_results = []
        for call in response.tool_calls:
            result = execute_tool(call.name, call.input)   # despacha para tools.py ou predictor.py
            tool_results.append({ "tool": call.name, "result": result })

        # Devolve os resultados para a LLM gerar a resposta final
        response = llm_client.call(
            messages=messages + [response.raw, tool_results],
            tools=tools
        )

    return {
        "text": response.text,
        "has_prediction": "run_prediction_engine" in response.tools_used,
        "prediction": response.prediction_data   # None se NN não foi acionada
    }
```

---

## 4. Tools disponíveis para a LLM

Todas as tools são funções Python em `tools.py` cujos schemas são declarados para a LLM. Ela chama o que julgar necessário, podendo chamar múltiplas tools em paralelo.

```python
# tools.py

def get_team_stats(team_name: str, last_n: int = 10, league: str = None) -> dict:
    """Stats agregadas recentes: aproveitamento, gols, cartões, escanteios,
    posse média, chutes, forma dos últimos N jogos."""

def get_player_stats(player_name: str, league: str = None, season: str = "current") -> dict:
    """Stats do jogador na temporada: gols, assistências, cartões,
    minutos jogados, passes certos %, duelos ganhos."""

def get_h2h(team_a: str, team_b: str, last_n: int = 10) -> dict:
    """Histórico de confrontos diretos: vitórias, empates, médias de
    gols / cartões / escanteios por jogo entre os dois times."""

def get_league_table(league: str, season: str = "current") -> list:
    """Tabela classificatória: posição, pontos, saldo de gols,
    aproveitamento, gols marcados e sofridos."""

def get_team_schedule(team_name: str, upcoming: bool = True) -> list:
    """Próximos jogos ou partidas recentes de um time."""

def run_prediction_engine(home_team: str, away_team: str, league: str) -> dict:
    """[ACIONA A NN] Retorna probabilidades para resultado,
    cartões amarelos e escanteios da partida."""
```

---

## 5. LLM Client — Tools disponíveis para a LLM

```python
# tools.py — funções que a LLM pode chamar (Tool Use pattern)

def get_team_stats(team_name: str, last_n: int = 10) -> dict:
    """
    Retorna estatísticas recentes de um time:
    aproveitamento, gols marcados/sofridos, cartões,
    escanteios, forma recente (últimos N jogos)
    """

def get_player_stats(player_name: str, league: str = None) -> dict:
    """
    Retorna estatísticas de um jogador na temporada atual:
    gols, assistências, cartões, minutos jogados, rating médio
    """

def get_h2h(team_a: str, team_b: str, last_n: int = 10) -> dict:
    """
    Retorna histórico de confrontos diretos:
    vitórias de cada lado, empates, médias de gols/cartões/escanteios
    """

def get_league_table(league: str, season: str = "current") -> list:
    """
    Retorna tabela classificatória com pontos, saldo de gols,
    aproveitamento, gols marcados/sofridos
    """

def get_team_schedule(team_name: str, upcoming: bool = True) -> list:
    """
    Retorna os próximos jogos ou jogos recentes do time
    """

def run_prediction_engine(home_team: str, away_team: str, league: str) -> dict:
    """
    Aciona o predictor.py com a rede neural.
    Retorna: probabilidades de resultado, cartões e escanteios
    """
```

---

## 6. System Prompt da LLM (Especialista de Futebol)

```
Você é OscaBet, um especialista em futebol com acesso a dados históricos
detalhados de partidas, times e jogadores de múltiplas ligas.

Seu papel:
- Responder perguntas sobre futebol com precisão, usando os dados fornecidos.
- Dar opiniões concisas e embasadas, citando estatísticas relevantes.
- Quando houver previsão da rede neural disponível, incorporá-la na resposta
  apresentando os números de forma clara (não como certeza, mas como análise).
- Ser direto e objetivo: máximo de 3-4 parágrafos por resposta.
- Nunca inventar estatísticas. Se não tiver dados suficientes, diga claramente.
- Ao apresentar previsões, lembre o usuário que são probabilidades, não certezas.

Estilo de resposta:
- Tom: analítico, confiante, como um comentarista esportivo experiente.
- Estrutura: contexto → dados relevantes → opinião/previsão.
- Quando houver previsão da NN: retornar também um bloco JSON estruturado
  para renderização do PredictionCard no frontend.
```

---

## 7. Sistema de Auto-Atualização do Banco de Dados

### 7.1 Fluxo completo de atualização

```
TRIGGER (manual via /api/update ou agendado pelo scheduler.py)
        │
        ▼
┌─────────────────┐
│   backup.py     │
│                 │
│ Cria snapshot   │
│ timestampado em │
│ data/backups/   │
│ YYYY-MM-DD_HH/  │
└────────┬────────┘
         │ Backup confirmado
         ▼
┌─────────────────┐
│  run_scraper.py │
│                 │
│ Coleta apenas   │
│ partidas NOVAS  │
│ (desde último   │
│ scraping)       │
└────────┬────────┘
         │ Novos dados coletados em data/raw/
         ▼
┌─────────────────┐
│ preprocessor.py │
│                 │
│ Recalcula       │
│ features para   │
│ data/processed/ │
└────────┬────────┘
         │ Features atualizadas
         ▼
┌─────────────────┐
│   db_swap.py    │
│                 │
│ Substitui banco │
│ ativo pelo novo │
│ (atomic swap)   │
└────────┬────────┘
         │ Banco atualizado
         ▼
┌─────────────────┐
│  update_manager │
│                 │
│ Registra log:   │
│ data, qtd novos │
│ matches, status │
└─────────────────┘
```

### 7.2 Scraping incremental (apenas dados novos)

O `storage.py` verifica o `scraper_log.json` para saber a data da última coleta e busca somente partidas posteriores a ela, evitando reprocessar o banco inteiro.

```python
# Exemplo de lógica em collector.py
def collect_since_last_update(league_id, season_id, last_match_date):
    """
    Percorre as páginas de eventos do mais recente para o mais antigo.
    Para quando encontra uma partida anterior a last_match_date.
    """
```

### 7.3 Backup automático

```
data/backups/
├── 2025-05-15_08-00/
│   ├── matches.csv          (cópia completa)
│   └── match_stats.csv      (cópia completa)
├── 2025-05-20_08-00/
│   ├── matches.csv
│   └── match_stats.csv
└── latest -> 2025-05-20_08-00/   (symlink para o mais recente)
```

Regra de retenção: mantém os **5 backups mais recentes**, deleta os mais antigos automaticamente.

### 7.4 Agendamento (scheduler.py)

```python
# scheduler.py — usa APScheduler (instalado via pip)
# Execução padrão: toda segunda-feira às 03:00 (pós-rodada do fim de semana)

from apscheduler.schedulers.background import BackgroundScheduler
from update_manager import run_full_update

scheduler = BackgroundScheduler()
scheduler.add_job(
    func=run_full_update,
    trigger="cron",
    day_of_week="mon",
    hour=3,
    minute=0,
    id="weekly_update"
)
```

---

## 8. Exemplos de Interação no Chat

**Modo chat — pergunta geral (campo de texto):**
```
Usuário: "Como está a forma recente do Atlético-MG?"

Agente: "O Atlético-MG vive uma fase irregular. Nos últimos 10 jogos,
o time soma 5 vitórias, 2 empates e 3 derrotas (55% de aproveitamento),
com média de 1.7 gols marcados e 1.4 sofridos por partida. O ponto de
atenção está na defesa: em 4 dos últimos 5 jogos o time cedeu ao menos
um gol. O ataque, por outro lado, marcou em 9 dos 10 últimos jogos."
```

**Modo previsão — botão "Prever Partida" com Flamengo x Grêmio / Brasileirão:**
```
Agente: "O Flamengo parte como favorito claro jogando no Maracanã, onde
acumula 78% de aproveitamento nos últimos 12 jogos. O Grêmio chega em
boa fase fora de casa (4 vitórias nos últimos 6), mas o histórico favorece
o mandante: 6 vitórias para o Flamengo nos últimos 10 confrontos diretos.

Em relação aos escanteios, ambos os times estão entre os que mais geram
oportunidades na borda — média combinada de 11.2 por jogo nesses confrontos.

[PredictionCard — renderizado pelo frontend]
Resultado:  Flamengo 64% | Empate 22% | Grêmio 14%
Escanteios: Baixo 12%    | Médio 48%  | Alto 40%
Cartões:    Baixo 21%    | Médio 45%  | Alto 34%
```

---

## 9. Features do Modelo Neural

Todas as features são calculadas como **médias das últimas N partidas** de cada time antes da data da partida a ser prevista. O `preprocessor.py` garante que nenhuma informação posterior à data do jogo vaze para o cálculo.

### 9.1 Features de Resultado
| Feature | Descrição |
|---|---|
| `home_win_rate` | % vitórias do mandante (últimos N jogos) |
| `away_win_rate` | % vitórias do visitante |
| `home_draw_rate` | % empates do mandante |
| `away_draw_rate` | % empates do visitante |
| `home_goals_scored_avg` | Média de gols marcados (mandante) |
| `away_goals_scored_avg` | Média de gols marcados (visitante) |
| `home_goals_conceded_avg` | Média de gols sofridos (mandante) |
| `away_goals_conceded_avg` | Média de gols sofridos (visitante) |
| `home_form_5` | Pontuação de forma últimos 5 jogos (3=V, 1=E, 0=D) |
| `away_form_5` | Pontuação de forma últimos 5 jogos |
| `home_position_norm` | Posição na tabela normalizada [0,1] |
| `away_position_norm` | Posição na tabela normalizada [0,1] |
| `h2h_home_win_rate` | % vitórias do mandante no confronto direto |
| `h2h_draw_rate` | % empates no confronto direto |

### 9.2 Features de Cartões Amarelos
| Feature | Descrição |
|---|---|
| `home_yellow_avg` | Média de amarelos por jogo (mandante) |
| `away_yellow_avg` | Média de amarelos por jogo (visitante) |
| `home_yellow_conceded_avg` | Média de amarelos que adversários recebem contra o mandante |
| `away_yellow_conceded_avg` | Média de amarelos que adversários recebem contra o visitante |
| `h2h_yellow_avg` | Média de amarelos totais nos confrontos diretos |
| `home_fouls_avg` | Média de faltas cometidas (mandante) |
| `away_fouls_avg` | Média de faltas cometidas (visitante) |
| `home_fouls_suffered_avg` | Média de faltas sofridas (mandante) |
| `away_fouls_suffered_avg` | Média de faltas sofridas (visitante) |
| `league_yellow_avg` | Média de amarelos da liga na temporada atual |

### 9.3 Features de Escanteios
| Feature | Descrição |
|---|---|
| `home_corners_for_avg` | Média de escanteios a favor (mandante) |
| `away_corners_for_avg` | Média de escanteios a favor (visitante) |
| `home_corners_against_avg` | Média de escanteios contra (mandante) |
| `away_corners_against_avg` | Média de escanteios contra (visitante) |
| `h2h_corners_avg` | Média de escanteios totais nos confrontos diretos |
| `league_corners_avg` | Média de escanteios da liga na temporada atual |

### 9.4 Features de Posse e Controle de Jogo
| Feature | Descrição |
|---|---|
| `home_possession_avg` | Média de posse de bola % (mandante) |
| `away_possession_avg` | Média de posse de bola % (visitante) |
| `home_passes_avg` | Média de passes totais (mandante) |
| `away_passes_avg` | Média de passes totais (visitante) |
| `home_passes_accurate_pct` | % de passes certos (mandante) |
| `away_passes_accurate_pct` | % de passes certos (visitante) |

### 9.5 Features de Criação e Finalização
| Feature | Descrição |
|---|---|
| `home_shots_avg` | Média de chutes totais (mandante) |
| `away_shots_avg` | Média de chutes totais (visitante) |
| `home_shots_on_target_avg` | Média de chutes no gol (mandante) |
| `away_shots_on_target_avg` | Média de chutes no gol (visitante) |
| `home_shots_off_target_avg` | Média de chutes fora (mandante) |
| `away_shots_off_target_avg` | Média de chutes fora (visitante) |
| `home_shots_blocked_avg` | Média de chutes bloqueados (mandante) |
| `away_shots_blocked_avg` | Média de chutes bloqueados (visitante) |
| `home_big_chances_avg` | Média de grandes chances criadas (mandante) |
| `away_big_chances_avg` | Média de grandes chances criadas (visitante) |
| `home_big_chances_missed_avg` | Média de grandes chances perdidas (mandante) |
| `away_big_chances_missed_avg` | Média de grandes chances perdidas (visitante) |

### 9.6 Features de Duelos e Pressão
| Feature | Descrição |
|---|---|
| `home_tackles_avg` | Média de desarmes (mandante) |
| `away_tackles_avg` | Média de desarmes (visitante) |
| `home_interceptions_avg` | Média de interceptações (mandante) |
| `away_interceptions_avg` | Média de interceptações (visitante) |
| `home_aerial_duels_won_pct` | % duelos aéreos ganhos (mandante) |
| `away_aerial_duels_won_pct` | % duelos aéreos ganhos (visitante) |
| `home_dribbles_avg` | Média de dribles bem-sucedidos (mandante) |
| `away_dribbles_avg` | Média de dribles bem-sucedidos (visitante) |
| `home_clearances_avg` | Média de cortes defensivos (mandante) |
| `away_clearances_avg` | Média de cortes defensivos (visitante) |

### 9.7 Features Adicionais
| Feature | Descrição |
|---|---|
| `home_offsides_avg` | Média de impedimentos (mandante) |
| `away_offsides_avg` | Média de impedimentos (visitante) |
| `home_saves_avg` | Média de defesas do goleiro (mandante) |
| `away_saves_avg` | Média de defesas do goleiro (visitante) |
| `home_xg_avg` | Média de xG — gols esperados (mandante) |
| `away_xg_avg` | Média de xG — gols esperados (visitante) |
| `home_red_cards_avg` | Média de cartões vermelhos (mandante) |
| `away_red_cards_avg` | Média de cartões vermelhos (visitante) |

> **Total: ~60 features.** O preprocessor aplica normalização min-max por liga para evitar que diferenças de escala entre campeonatos distorçam o modelo.

---

## 10. Metodologia de Treinamento — Controle Temporal

### 10.1 Conceito: janela de treino e janela de validação

O modelo é treinado exclusivamente com jogos **anteriores** à data de corte configurada. Os jogos após essa data servem como **conjunto de validação real** — o modelo nunca os viu durante o treino, e os resultados reais são usados para medir a qualidade das previsões.

```
Linha do tempo dos dados:

  ├─────────────────────────────────┤──────────────────┤
  │         TREINO                  │    VALIDAÇÃO     │
  │  (modelo aprende padrões)       │  (prova real)    │
  │                                 │                  │
  01/2020                     30/06/2025         hoje

  ↑ configurável em .env       ↑ TRAIN_CUTOFF_DATE
```

### 10.2 Configuração no `.env`

```env
# Janela de treinamento
TRAIN_START_DATE=2020-01-01       # início do período de treino
TRAIN_CUTOFF_DATE=2025-06-30      # data de corte: treina ATÉ aqui

# Janela de validação (do corte até os dados mais recentes no banco)
# Gerada automaticamente — não precisa configurar
```

### 10.3 Como o `trainer.py` aplica o corte

```python
# trainer.py

def build_train_val_split(features_df, targets_df, cutoff_date: str):
    """
    Divide os dados respeitando a ordem temporal.
    Nunca embaralha antes do corte — isso preserva a causalidade.
    """
    cutoff = pd.to_datetime(cutoff_date)

    train_mask = features_df["match_date"] <  cutoff
    val_mask   = features_df["match_date"] >= cutoff

    X_train = features_df[train_mask].drop(columns=["match_date", "match_id"])
    X_val   = features_df[val_mask].drop(columns=["match_date", "match_id"])

    y_train = targets_df[train_mask]
    y_val   = targets_df[val_mask]

    return X_train, X_val, y_train, y_val
```

> **Por que não embaralhar?** Em time-series esportivas, embaralhar antes do corte causaria **data leakage**: o modelo aprenderia padrões de jogos futuros para prever jogos passados, inflando artificialmente as métricas de treino e produzindo um modelo inútil na prática.

### 10.4 Processo de treino no notebook

```
01_exploracao_dados.ipynb
  └── Análise da distribuição temporal dos dados
  └── Histogramas de cartões e escanteios por liga
  └── Verificação de partidas sem estatísticas (dados faltantes)

02_preprocessamento.ipynb
  └── Aplicar janela deslizante de N jogos por time
  └── Calcular todas as features da seção 9
  └── Garantir que nenhuma feature usa dados após a data da partida
  └── Normalizar por liga
  └── Exportar features.csv e targets.csv com match_date preservado

03_treinamento.ipynb
  └── Carregar features.csv e targets.csv
  └── Chamar build_train_val_split(cutoff_date=TRAIN_CUTOFF_DATE)
  └── Treinar rede neural por N épocas
  └── Avaliar no conjunto de validação (resultados reais)
  └── Salvar pesos em agent/models/oscabet_nn_v1.pt
  └── Registrar métricas: acurácia por alvo, log-loss, calibração
```

### 10.5 Métricas de avaliação do treino

| Métrica | O que mede |
|---|---|
| **Acurácia (resultado)** | % de previsões corretas de H/D/A no conjunto de validação |
| **Acurácia (cartões)** | % corretas de baixo/médio/alto no conjunto de validação |
| **Acurácia (escanteios)** | % corretas de baixo/médio/alto no conjunto de validação |
| **Log-loss por alvo** | Qualidade das probabilidades (não só do label mais provável) |
| **Calibration plot** | Probabilidades previstas vs. frequências reais (ex: jogos com 70% de chance de H — quantos % realmente terminaram em H?) |
| **Cobertura temporal** | Quantos jogos da janela de validação foram corretamente cobertos |

---

## 11. Arquitetura da Rede Neural (Multi-task)

```
Input Layer (~60 features normalizadas por liga)
        │
        ▼
Dense(256, ReLU) + BatchNorm + Dropout(0.3)
        │
        ▼
Dense(128, ReLU) + BatchNorm + Dropout(0.2)
        │
        ▼
Dense(64, ReLU)
        │
   ┌────┴──────────────┬────────────────┐
   ▼                   ▼                ▼
Dense(32, ReLU)   Dense(32, ReLU)  Dense(32, ReLU)
   │                   │                │
   ▼                   ▼                ▼
Softmax(3)         Softmax(3)       Softmax(3)
[H / D / A]    [baixo/médio/alto] [baixo/médio/alto]
  Resultado        Cartões          Escanteios

Loss total = 1.0 * loss_resultado
           + 0.8 * loss_cartoes
           + 0.8 * loss_escanteios

Treinamento:
  - Otimizador: Adam (lr=1e-3, weight_decay=1e-4)
  - Scheduler: ReduceLROnPlateau (patience=10)
  - Early stopping: patience=20 épocas sem melhora no val_loss
  - Dados de treino: partidas ANTES de TRAIN_CUTOFF_DATE
  - Dados de validação: partidas APÓS TRAIN_CUTOFF_DATE (resultados reais)
```

---

## 11. Endpoints da API Flask

| Método | Rota | Descrição |
|---|---|---|
| `POST` | `/api/chat` | Mensagem livre → LLM decide quais tools chamar, incluindo a NN quando necessário |
| `POST` | `/api/predict` | Previsão direta (sem chat) para um par de times |
| `GET` | `/api/history/:team` | Histórico de partidas de um time |
| `POST` | `/api/update` | Dispara backup + scraping + atualização do banco |
| `GET` | `/api/update/status` | Retorna status da última atualização |
| `GET` | `/api/health` | Health check |

**Corpo do `/api/chat`:**
```json
{
  "message": "Flamengo x Grêmio no domingo, quem vence?",
  "conversation_id": "abc123",
  "history": [
    { "role": "user",      "content": "mensagem anterior" },
    { "role": "assistant", "content": "resposta anterior" }
  ]
}
```

**Resposta do `/api/chat`:**
```json
{
  "text": "O Flamengo parte como favorito...",
  "has_prediction": true,
  "prediction": {
    "match": "Flamengo vs Grêmio",
    "result": { "label": "H", "confidence": 0.64, "probs": {...} },
    "yellow_cards": { "label": "medium", "confidence": 0.45, "probs": {...} },
    "corners": { "label": "high", "confidence": 0.40, "probs": {...} },
    "explanation": "..."
  }
}
```

---

## 12. Scraper — Arquitetura Detalhada

### 12.1 Endpoints do Sofascore utilizados

```python
BASE = "https://api.sofascore.com/api/v1"

SEASONS      = f"{BASE}/tournament/{{league_id}}/seasons"
EVENTS_LAST  = f"{BASE}/tournament/{{league_id}}/season/{{season_id}}/events/last/{{page}}"
EVENTS_NEXT  = f"{BASE}/tournament/{{league_id}}/season/{{season_id}}/events/next/{{page}}"
MATCH_STATS  = f"{BASE}/event/{{match_id}}/statistics"
MATCH_DETAIL = f"{BASE}/event/{{match_id}}"
```

### 12.2 Headers anti-bloqueio (Cloudflare)

```python
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0.0.0 Safari/537.36",
    "Accept": "application/json",
    "Accept-Language": "pt-BR,pt;q=0.9",
    "Referer": "https://www.sofascore.com/",
    "Origin": "https://www.sofascore.com",
}
# Delay aleatório: 1.5–3.0 segundos entre requisições
```

### 12.3 Ligas configuradas

```python
LEAGUES = {
    "brasileirao_a"   : 325,
    "brasileirao_b"   : 390,
    "copa_brasil"     : 162,
    "libertadores"    : 384,
    "premier_league"  : 17,
    "la_liga"         : 8,
    "serie_a"         : 23,
    "bundesliga"      : 35,
    "ligue_1"         : 34,
    "champions_league": 7,
}

SEASONS_RANGE = ["24/25", "23/24", "22/23", "21/22", "20/21"]
```

### 12.4 Schema dos CSVs

**`matches.csv`** — uma linha por partida:
`match_id, league, season, date, home_team, away_team, home_team_id, away_team_id, home_score, away_score, result, stats_collected`

**`match_stats.csv`** — estatísticas completas por partida (tudo que o Sofascore disponibiliza):

| Coluna | Tipo | Descrição |
|---|---|---|
| `match_id` | int | FK para matches.csv |
| `home_possession` | float | Posse de bola mandante (%) |
| `away_possession` | float | Posse de bola visitante (%) |
| `home_shots` | int | Chutes totais mandante |
| `away_shots` | int | Chutes totais visitante |
| `home_shots_on_target` | int | Chutes no gol mandante |
| `away_shots_on_target` | int | Chutes no gol visitante |
| `home_shots_off_target` | int | Chutes fora mandante |
| `away_shots_off_target` | int | Chutes fora visitante |
| `home_shots_blocked` | int | Chutes bloqueados mandante |
| `away_shots_blocked` | int | Chutes bloqueados visitante |
| `home_big_chances` | int | Grandes chances criadas mandante |
| `away_big_chances` | int | Grandes chances criadas visitante |
| `home_big_chances_missed` | int | Grandes chances perdidas mandante |
| `away_big_chances_missed` | int | Grandes chances perdidas visitante |
| `home_corners` | int | Escanteios mandante |
| `away_corners` | int | Escanteios visitante |
| `total_corners` | int | Total de escanteios |
| `home_yellow_cards` | int | Cartões amarelos mandante |
| `away_yellow_cards` | int | Cartões amarelos visitante |
| `total_yellow_cards` | int | Total de cartões amarelos |
| `home_red_cards` | int | Cartões vermelhos mandante |
| `away_red_cards` | int | Cartões vermelhos visitante |
| `home_fouls` | int | Faltas cometidas mandante |
| `away_fouls` | int | Faltas cometidas visitante |
| `home_offsides` | int | Impedimentos mandante |
| `away_offsides` | int | Impedimentos visitante |
| `home_passes` | int | Passes totais mandante |
| `away_passes` | int | Passes totais visitante |
| `home_passes_accurate` | int | Passes certos mandante |
| `away_passes_accurate` | int | Passes certos visitante |
| `home_tackles` | int | Desarmes mandante |
| `away_tackles` | int | Desarmes visitante |
| `home_interceptions` | int | Interceptações mandante |
| `away_interceptions` | int | Interceptações visitante |
| `home_clearances` | int | Cortes defensivos mandante |
| `away_clearances` | int | Cortes defensivos visitante |
| `home_aerial_duels_won` | int | Duelos aéreos ganhos mandante |
| `away_aerial_duels_won` | int | Duelos aéreos ganhos visitante |
| `home_aerial_duels_total` | int | Duelos aéreos totais mandante |
| `away_aerial_duels_total` | int | Duelos aéreos totais visitante |
| `home_dribbles_successful` | int | Dribles bem-sucedidos mandante |
| `away_dribbles_successful` | int | Dribles bem-sucedidos visitante |
| `home_saves` | int | Defesas do goleiro mandante |
| `away_saves` | int | Defesas do goleiro visitante |
| `home_xg` | float | Expected goals mandante (quando disponível) |
| `away_xg` | float | Expected goals visitante (quando disponível) |

> **Nota:** Nem todos os campos estão disponíveis em todas as ligas/temporadas. O scraper salva `null` para campos ausentes. O preprocessor trata os nulos antes de calcular as features.

---

## 13. Setup do Ambiente — Passo a Passo

### Pré-requisitos
VS Code, Python 3.x, pip, Homebrew, Node.js + npm, MiniConda (verificar abaixo)

### 13.1 Verificar / instalar MiniConda

```bash
conda --version

# Se não instalado:
brew install --cask miniconda
conda init zsh
# Feche e reabra o terminal
```

### 13.2 Clonar repositório e criar estrutura

```bash
git clone https://github.com/SEU_USUARIO/oscabet-agent.git
cd oscabet-agent

mkdir -p data/raw data/processed data/backups
mkdir -p scraper/src
mkdir -p agent/src agent/notebooks agent/models agent/tests
mkdir -p updater
mkdir -p api/routes
mkdir -p web/src/pages web/src/components
```

### 13.3 Ambiente Conda do Scraper

```bash
cd scraper/
conda env create -f environment.yml
conda activate oscabet-scraper
```

**`scraper/environment.yml`:**
```yaml
name: oscabet-scraper
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - pandas
  - numpy
  - pip
  - pip:
    - requests
    - tqdm
    - python-dotenv
```

### 13.4 Ambiente Conda do Agent + API

```bash
cd ../agent/
conda env create -f environment.yml
conda activate oscabet

python -m ipykernel install --user --name=oscabet --display-name "Python (oscabet)"
```

**`agent/environment.yml`:**
```yaml
name: oscabet
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.11
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - jupyter
  - ipykernel
  - pip
  - pip:
    - torch
    - flask
    - flask-cors
    - python-dotenv
    - anthropic          # Claude API — LLM do agente
    - apscheduler        # Agendamento do updater
```

### 13.5 Frontend React

```bash
# Na raiz (somente primeira vez):
npm create vite@latest web -- --template react
cd web && npm install && npm run dev
# http://localhost:5173
```

### 13.6 Variáveis de ambiente

**`.env.example`:**
```env
# Flask
FLASK_ENV=development
FLASK_PORT=5000

# Claude API (LLM)
ANTHROPIC_API_KEY=sk-ant-...

# Paths
DATA_PATH=../data/processed/
RAW_DATA_PATH=../data/raw/
BACKUP_PATH=../data/backups/
MODEL_PATH=../agent/models/oscabet_nn_v1.pt

# Scraper
SCRAPER_DELAY_MIN=1.5
SCRAPER_DELAY_MAX=3.0

# Agent
DEFAULT_WINDOW=10
MIN_GAMES_REQUIRED=3
CONFIDENCE_THRESHOLD=0.5

# Treinamento — janela temporal
TRAIN_START_DATE=2020-01-01
TRAIN_CUTOFF_DATE=2025-06-30      # treina ATÉ aqui; valida com o restante


UPDATE_SCHEDULE_DAY=mon
UPDATE_SCHEDULE_HOUR=3

# Thresholds de classificação
CORNERS_LOW_MAX=7
CORNERS_HIGH_MIN=12
YELLOW_LOW_MAX=3
YELLOW_HIGH_MIN=6
```

---

## 14. Fluxo de Desenvolvimento (ordem recomendada)

```
[1] Scraper
    └── client.py → endpoints.py → match_scraper.py → collector.py → storage.py
    └── Gerar: data/raw/matches.csv + match_stats.csv

[2] Explorar dados
    └── 01_exploracao_dados.ipynb

[3] Preprocessador + features
    └── preprocessor.py + 02_preprocessamento.ipynb

[4] Rede neural multi-alvo
    └── model.py + trainer.py + 03_treinamento.ipynb

[5] Predictor + Explainer
    └── predictor.py: 3 alvos
    └── explainer.py: justificativa por alvo

[6] Router
    └── router.py: campeonato → subset de dados

[7] Tools e Orchestrator
    └── tools.py: funções que a LLM pode chamar (get_team_stats, get_h2h, etc.)
    └── orchestrator.py: modo "chat" → só LLM; modo "predict" → NN + LLM

[8] LLM Client + Orchestrator
    └── llm_client.py: wrapper Claude API
    └── orchestrator.py: monta o plano e coordena as engines

[9] Updater
    └── backup.py → db_swap.py → update_manager.py → scheduler.py

[10] API Flask
    └── /api/chat conectando tudo
    └── /api/update conectando o updater

[11] Frontend React (gerado por IA)
    └── ChatWindow + MessageBubble + PredictionCard + UpdatePanel

[12] Testes e documentação
    └── 04_testes_agente.ipynb — ≥ 5 cenários
    └── README.md com instruções de execução
```

---

## 15. .gitignore

```gitignore
__pycache__/
*.pyc
.env
*.egg-info/
.conda/
.ipynb_checkpoints/
agent/models/*.pt
web/node_modules/
web/dist/
data/raw/
data/backups/
.DS_Store
```

---

## 16. Checklist de Entrega

### Etapa 1
- [ ] Problema delimitado (chat especialista + 3 mercados de previsão)
- [ ] Diagrama de arquitetura das duas engines
- [ ] Mockup do chat com PredictionCard
- [ ] Padrões de IA agêntica identificados e mapeados
- [ ] Features e métricas definidas

### Etapa 2
- [ ] Scraper funcional e incremental
- [ ] Sistema de backup e auto-atualização
- [ ] ≥ 3 funções Python por módulo
- [ ] Condicionais em `intent_classifier.py`, `router.py`, `predictor.py`
- [ ] Listas e dicionários em `tools.py`, `explainer.py`, `data_loader.py`
- [ ] Modelo multi-alvo treinado com corte temporal configurável (`TRAIN_CUTOFF_DATE`)
- [ ] Validação feita com jogos reais posteriores ao corte (sem data leakage)
- [ ] Calibration plot e métricas por alvo registradas no notebook
- [ ] Orchestrator com agentic loop: LLM decide tools, executor Python as chama
- [ ] `run_prediction_engine` registrada como tool com description clara para a LLM
- [ ] Chat respondendo perguntas gerais e acionando NN automaticamente quando pertinente
- [ ] Frontend exibindo PredictionCard apenas quando NN foi acionada
- [ ] Chat respondendo perguntas gerais e de previsão
- [ ] `/api/update` funcional com backup automático
- [ ] Frontend com chat e PredictionCard
- [ ] ≥ 5 cenários de teste documentados
- [ ] README com instruções completas