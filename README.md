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

A interface (`web/`) é um chat React (via CDN, sem build) servido pelo Flask.
O backend só **consome** o modelo (`oscabet_nn_v1.pt`) via `orchestrator`/`predictor` —
re-treinar a rede apenas regenera o `.pt` e o app passa a usá-lo automaticamente.

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
| `run_prediction_engine` | Previsão da rede neural para uma partida futura |

---

## Próximas Etapas

- [x] `app.py` — API Flask com endpoint `/api/chat`
- [x] Frontend React — ChatWindow + PredictionCard (em `web/`)
- [ ] Scraper Sofascore — substituição dos dados sintéticos
- [ ] Auto-atualização do banco de dados (APScheduler)

---

## Tecnologias

`Python 3.11` · `PyTorch` · `Pandas` · `scikit-learn` · `Ollama` · `Flask` · `React`
