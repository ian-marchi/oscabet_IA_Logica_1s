# OscaBet Agent 🤖⚽

> **Trabalho de IA Agêntica — 1º Período**  
> Especialista de futebol com chat em linguagem natural, previsões por rede neural e auto-atualização de banco de dados.

---

## Visão Geral

O OscaBet é um agente de IA que responde perguntas sobre futebol em linguagem natural, combinando dados históricos de partidas com uma rede neural multi-alvo capaz de prever:

- **Resultado** — Vitória Mandante / Empate / Vitória Visitante
- **Cartões Amarelos** — Over/Under 4.5
- **Escanteios** — Over/Under 9.5

---

## Ligas Suportadas

| Liga | Identificador |
|---|---|
| Brasileirão Série A | `brasileirao_a` |
| Premier League | `premier_league` |
| La Liga | `la_liga` |

---

## Estrutura do Projeto

```
oscabet_IA_Logica_1s/
├── .env                        # Variáveis de ambiente (não versionar)
├── .env.example                # Template do .env
├── generate_data.py            # Gerador de dados sintéticos
├── create_notebooks.py         # Criador dos notebooks
│
├── data/
│   ├── raw/
│   │   ├── matches.csv         # Partidas brutas
│   │   └── match_stats.csv     # Estatísticas por partida
│   └── processed/
│       ├── features.csv        # Features rolling-window (~70 colunas)
│       └── targets.csv         # Targets: resultado, cartões, escanteios
│
└── agent/
    ├── models/
    │   ├── oscabet_nn_v1.pt    # Modelo treinado (PyTorch)
    │   └── oscabet_nn_v1_meta.json
    │
    ├── notebooks/
    │   ├── 01_exploracao_dados.ipynb
    │   ├── 02_preprocessamento.ipynb
    │   ├── 03_treinamento.ipynb
    │   └── 04_testes_agente.ipynb
    │
    └── src/
        ├── data_loader.py      # Acesso centralizado aos CSVs
        ├── predictor.py        # Inferência da rede neural
        ├── tools.py            # 5 tools chamáveis pela LLM
        ├── llm_client.py       # Wrapper Ollama Cloud
        ├── orchestrator.py     # Agentic loop principal
        └── test_tools.py       # Testes das tools sem LLM
```

---

## Instalação

### 1. Pré-requisitos

- macOS com [Homebrew](https://brew.sh)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
- Conta em [ollama.com](https://ollama.com) com API key

### 2. Instalar Ollama

```bash
brew install ollama
ollama signin
```

### 3. Criar o ambiente Conda

```bash
conda create -n oscabet python=3.11 numpy pandas scikit-learn \
  matplotlib seaborn jupyter ipykernel tqdm -c conda-forge

conda activate oscabet

pip install torch flask flask-cors python-dotenv ollama \
  apscheduler nbformat
```

### 4. Registrar kernel do Jupyter

```bash
python -m ipykernel install --user --name=oscabet \
  --display-name "Python (oscabet)"
```

### 5. Configurar variáveis de ambiente

```bash
cp .env.example .env
# Edita o .env e preenche OLLAMA_API_KEY com sua chave
# https://ollama.com/settings/keys
```

---

## Como Usar

### Gerar dados e treinar o modelo

ollama serve

```bash
conda activate oscabet
cd oscabet_IA_Logica_1s

# 1. Gera os dados sintéticos
python generate_data.py

# 2. Abre os notebooks em ordem
cd agent
jupyter notebook notebooks/
# Rode: 01 → 02 → 03 → 04
```

> **Modelo de produção (ensemble):** o `.pt` usado pelo app é um **ensemble de
> seeds** (média das probabilidades — mais estável e calibrado). Gere/atualize com:
> ```bash
> python agent/train_ensemble.py        # treina 5 seeds e salva oscabet_nn_v1.pt
> ```
> O notebook 03 treina um modelo único (didático); o `predictor.py` carrega ambos
> os formatos. A previsão de **resultado** ainda nomeia *Empate* em jogos
> equilibrados (ver `DRAW_MARGIN` no `.env`).

> Quando o banco real do Sofascore estiver disponível, substitua os arquivos
> em `data/raw/` e re-execute a partir do notebook 02.

### Testar as tools sem LLM

```bash
cd agent/src
python test_tools.py
```

### Rodar o agente em modo terminal

```bash
cd agent/src
python orchestrator.py
```

### Rodar a aplicação web (Flask + React)

```bash
conda activate oscabet      # IMPORTANTE: ativa o ambiente (carrega as DLLs do PyTorch)
python app.py
# abre http://localhost:5000
```

A interface (`web/`) é um React (via CDN, sem build) servido pelo Flask, com duas abas:
- **Chat** — converse com o agente (linguagem natural, tools, previsões).
- **Painel** — *Simulador* (escolha dois times → previsão na hora) e *Próxima rodada*
  (jogos futuros com previsão + apostas de valor quando há odds).

Endpoints extras do dashboard: `/api/leagues`, `/api/teams`, `/api/predict`, `/api/upcoming`.
O backend só **consome** o modelo (`oscabet_nn_v1.pt`) via `orchestrator`/`predictor` —
re-treinar a rede apenas regenera o `.pt` e o app passa a usá-lo automaticamente.

Testes automatizados: `python agent/src/test_suite.py` (13 testes do núcleo).

> **Windows:** rode sempre com o ambiente `oscabet` ativado. Sem ele, o `import torch`
> falha com `OSError: [WinError 127] ... shm.dll` (faltam as DLLs do `Library\bin`).

---

## Arquitetura da Rede Neural

```
Input (~71 features)
       │
  ┌────▼─────────────────────┐
  │  Dense(256) → BN → ReLU  │
  │  Dropout(0.30)            │  Backbone
  │  Dense(128) → BN → ReLU  │  compartilhado
  │  Dropout(0.20)            │
  │  Dense(64)  → ReLU        │
  └────┬──────────┬──────────┘
       │          │          │
  ┌────▼──┐  ┌───▼───┐  ┌───▼───┐
  │Head R │  │Head Y │  │Head C │
  │Dense32│  │Dense32│  │Dense32│
  │Soft(3)│  │Soft(2)│  │Soft(2)│
  └───────┘  └───────┘  └───────┘
  Resultado  Cartões   Escanteios
  H/D/A      O/U 4.5   O/U 9.5
```

**Features de entrada (~71):** rolling window dos últimos 10 jogos por time —
aproveitamento, gols, xG, posse, passes, chutes, escanteios, cartões, duelos,
H2H, posição na tabela e médias da liga.

---

## Tools do Agente

| Tool | Descrição |
|---|---|
| `get_team_stats` | Stats recentes de um time (forma, gols, cartões, escanteios) |
| `get_h2h` | Histórico de confrontos diretos entre dois times |
| `get_league_table` | Tabela classificatória completa de uma liga |
| `get_team_schedule` | Últimos jogos ou próximos jogos de um time |
| `run_prediction_engine` | Previsão da rede neural (suporta partidas fictícias entre ligas) |
| `get_value_bets` | Apostas de **valor** em jogos futuros: modelo × odds reais (EV ≥ piso) |

---

## Atualização automática (semanal)

O banco se atualiza sozinho a partir do Sofascore. Pipeline em `update_weekly.py`:
**coleta incremental** (só jogos novos, por liga) → **preprocessamento**
(`agent/preprocess.py`) → **retreino do ensemble** (`agent/train_ensemble.py`).

```bash
conda activate oscabet
python update_weekly.py                 # pipeline completo
python update_weekly.py --no-retrain    # só atualiza dados (sem retreinar)
python update_weekly.py --seasons 1     # varre só a temporada atual (mais rápido)
```

### Agendar toda semana

**Opção A — multiplataforma (APScheduler):** um processo que fica rodando.
```bash
python scheduler.py                     # default: segunda 04:00
python scheduler.py --day sun --hour 6  # personaliza
python scheduler.py --now               # teste: roda uma vez agora
```

**Opção B — agendador do SO (sobrevive a reboot, sem processo rodando):**

- **Windows (Task Scheduler):**
  ```powershell
  schtasks /create /tn "OscaBet Update" /tr "%CD%\run_update.bat" /sc weekly /d MON /st 04:00
  ```
- **macOS / Linux (cron):** `crontab -e` e adicione:
  ```cron
  0 4 * * 1 /caminho/para/oscabet_IA_Logica_1s/run_update.sh >> /caminho/logs/cron.log 2>&1
  ```
  (no macOS, dê permissão: `chmod +x run_update.sh`)

Logs em `logs/update_weekly.log`. O scraper é educado (rate limiting + cooldown
anti-ban), então a coleta leva alguns minutos.

---

## Apostas de valor (modelo × odds reais)

`agent/value_bets.py` busca jogos futuros + odds reais (Sofascore), roda o modelo e
recomenda apostas onde o valor esperado **EV = prob_modelo × odd − 1** passa de um
**piso** (default 5%). Cobre Resultado (1X2) e Escanteios O/U 9.5 (linha que casa com
as odds); Cartões só quando a casa oferece a linha 4.5. Sugere stake por Kelly fracionado.

```bash
conda activate oscabet
python agent/value_bets.py                          # brasileirao_a..., piso 5%
python agent/value_bets.py --league premier_league --floor 0.08
python agent/value_bets.py --event 15235586         # avalia um match_id (teste)
```

No chat do agente, basta perguntar *"quais as apostas de valor da rodada?"* — ele
chama a tool `get_value_bets`. Odds só aparecem perto do jogo; fora de rodada o
resultado vem vazio.

> ⚠️ Mede valor contra as odds disponíveis; não é garantia de lucro. Casas reais
> precificam força/forma (= nossas features), então a margem real é menor.

---

## Próximas Etapas

- [x] `app.py` — API Flask com endpoint `/api/chat`
- [x] Frontend React — ChatWindow + PredictionCard (em `web/`)
- [x] Scraper Sofascore — dados reais (`Banco de dados/scraper/`)
- [x] Auto-atualização semanal (`update_weekly.py` + `scheduler.py`)

---

## Tecnologias

`Python 3.11` · `PyTorch` · `Pandas` · `scikit-learn` · `Ollama` · `Flask` · `React`
