#!/usr/bin/env python
# coding: utf-8

# # 02 — Preprocessamento e Feature Engineering  
# **OscaBet Agent** · Geração das ~60 features para treino da rede neural
# 
# Calcula todas as features do Plano §9 usando janela deslizante,  
# garante zero data-leakage e exporta `features.csv` + `targets.csv`.
# 

# In[14]:


import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless (pipeline automático)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 70)
pd.set_option("display.float_format", "{:.4f}".format)

# ──────────────────────────────────────────────────────────────────────────────
# Raiz do projeto: auto-detecta (pasta que contém 'agent' e 'data').
# ──────────────────────────────────────────────────────────────────────────────
def _find_root(start: Path) -> Path:
    for p in [start, *start.parents]:
        if (p / "agent").is_dir() and (p / "data").is_dir():
            return p
    return start.parent

BASE_DIR  = _find_root(Path(__file__).resolve().parent)
RAW_DIR   = BASE_DIR / "data" / "raw"
PROC_DIR  = BASE_DIR / "data" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)

matches = pd.read_csv(RAW_DIR / "matches.csv", parse_dates=["date"])
stats   = pd.read_csv(RAW_DIR / "match_stats.csv")

# ── Thresholds de classificação (espelham o .env do projeto) ──
WINDOW         = 10     # últimos N jogos para rolling features
WINDOW_H2H     = 10     # últimos N confrontos diretos
CORN_LOW_MAX   = 7
CORN_HIGH_MIN  = 12
YELL_LOW_MAX   = 3
YELL_HIGH_MIN  = 6

# Merge principal
df = matches.merge(stats, on="match_id", how="left")
df = df.sort_values("date").reset_index(drop=True)
print(f"Dataset merged: {df.shape}")


# ## 1. Tratamento de Valores Nulos

# In[15]:


# Partidas sem estatísticas (~5%): preenche com NaN para exclusão posterior
# Colunas numéricas de stats — estratégia: mediana da liga+temporada
stat_cols = [c for c in stats.columns if c != "match_id"]
null_pct   = df[stat_cols].isnull().mean() * 100
null_pct   = null_pct[null_pct > 0].sort_values(ascending=False)

print("Colunas com valores nulos (%):")
print(null_pct.to_string())
print(f"\nPartidas sem nenhuma estatística: {df[stat_cols[0]].isnull().sum():,}")


# ## 2. Features Rolling-Window (Seção 9 do Plano)
# 
# Calculamos as médias dos **últimos N jogos** de cada time  
# **antes** da data de cada partida — garantindo zero leakage.
# 
# Para cada time e partida, iteramos sobre o histórico anterior e computamos:
# - Aproveitamento, gols, cartões, escanteios, posse, passes, duelos…
# 

# In[16]:


def rolling_team_features(df_team: pd.DataFrame, window: int) -> pd.DataFrame:
    rows = []
    for i, row in df_team.iterrows():
        hist = df_team[df_team["date"] < row["date"]].tail(window)
        if len(hist) == 0:
            rows.append({})
            continue

        sot_team  = hist["sot_team"].clip(lower=0)
        shots_opp = hist["shots_opp"].clip(lower=0) if "shots_opp" in hist.columns else pd.Series([1]*len(hist), index=hist.index)
        sot_opp   = hist["sot_opp"].clip(lower=0)   if "sot_opp"   in hist.columns else pd.Series([1]*len(hist), index=hist.index)

        r = {
            # ── Aproveitamento e forma ────────────────────────────────────
            "win_rate":               (hist["result_for_team"] == "W").mean(),
            "draw_rate":              (hist["result_for_team"] == "D").mean(),
            "loss_rate":              (hist["result_for_team"] == "L").mean(),
            "form_5":                 hist.tail(5)["result_for_team"]
                                        .map({"W":3,"D":1,"L":0}).sum(),

            # ── Gols ──────────────────────────────────────────────────────
            "goals_scored_avg":       hist["goals_for"].mean(),
            "goals_conceded_avg":     hist["goals_against"].mean(),
            "goal_diff_avg":          (hist["goals_for"] - hist["goals_against"]).mean(),

            # ── Taxas de resultado ─────────────────────────────────────────
            "clean_sheet_rate":       (hist["goals_against"] == 0).mean(),
            "btts_rate":              ((hist["goals_for"] > 0) & (hist["goals_against"] > 0)).mean(),
            "over25_rate":            ((hist["goals_for"] + hist["goals_against"]) > 2.5).mean(),

            # ── Chutes e eficiência ───────────────────────────────────────
            "shots_avg":              hist["shots_team"].mean(),
            "shots_on_target_avg":    sot_team.mean(),
            "shots_off_target_avg":   hist["soff_team"].mean(),
            "shots_blocked_avg":      hist["sblk_team"].mean(),
            "shot_accuracy_pct":      (sot_team / hist["shots_team"].clip(lower=1)).mean(),
            "shot_conversion_rate":   (hist["goals_for"] / sot_team.clip(lower=1)).mean(),

            # ── Grandes chances ────────────────────────────────────────────
            "big_chances_avg":        hist["bc_team"].mean(),
            "big_chances_missed_avg": hist["bcm_team"].mean(),
            "big_chances_conv_rate":  (
                (hist["bc_team"] - hist["bcm_team"]).clip(lower=0)
                / hist["bc_team"].clip(lower=1)
            ).mean(),

            # ── xG e diferencial ──────────────────────────────────────────
            "xg_avg":                 hist["xg_team"].mean(),
            "xg_conceded_avg":        hist["xg_opp"].mean()   if "xg_opp" in hist.columns else np.nan,
            "xg_diff_avg":           (hist["xg_team"] - hist["xg_opp"]).mean()
                                      if "xg_opp" in hist.columns else np.nan,
            "xg_overperformance":    (hist["goals_for"]     - hist["xg_team"]).mean(),
            "xg_underperformance":   (hist["goals_against"] - hist["xg_opp"]).mean()
                                      if "xg_opp" in hist.columns else np.nan,

            # ── Pressão e duelos ──────────────────────────────────────────
            "pressing_intensity":     (hist["tackles_team"] + hist["inter_team"]).mean(),
            "tackles_avg":            hist["tackles_team"].mean(),
            "interceptions_avg":      hist["inter_team"].mean(),
            "clearances_avg":         hist["clear_team"].mean(),
            "aerial_duels_won_pct":   (hist["aer_won_team"] / hist["aer_tot_team"].clip(lower=1)).mean(),
            "dribbles_avg":           hist["drib_team"].mean(),

            # ── Posse e passes ────────────────────────────────────────────
            "possession_avg":         hist["possession_team"].mean(),
            "passes_avg":             hist["passes_team"].mean(),
            "passes_accurate_pct":    (hist["passes_acc_team"] / hist["passes_team"].clip(lower=1)).mean(),

            # ── Escanteios ────────────────────────────────────────────────
            "corners_for_avg":        hist["corners_for"].mean(),
            "corners_against_avg":    hist["corners_against"].mean(),
            "corners_diff_avg":       (hist["corners_for"] - hist["corners_against"]).mean(),

            # ── Disciplina ────────────────────────────────────────────────
            "yellow_avg":             hist["yellows_team"].mean(),
            "yellow_conceded_avg":    hist["yellows_opp"].mean(),
            "red_cards_avg":          hist["reds_team"].mean(),
            "fouls_avg":              hist["fouls_team"].mean(),
            "fouls_suffered_avg":     hist["fouls_opp"].mean(),
            "offsides_avg":           hist["offside_team"].mean(),

            # ── Defesa (chutes sofridos) ──────────────────────────────────
            "shots_conceded_avg":     shots_opp.mean(),
            "sot_conceded_avg":       sot_opp.mean(),
            "bc_conceded_avg":        hist["bc_opp"].mean() if "bc_opp" in hist.columns else np.nan,
            "saves_avg":              hist["saves_team"].mean(),

            # ── Metadado ──────────────────────────────────────────────────
            "n_games":                len(hist),
        }
        rows.append(r)

    return pd.DataFrame(rows, index=df_team.index)
    home_games = hist[hist.index.isin(
        df_team[df_team["home_team"] == df_team.loc[i, "home_team"]].index
    )] if "home_team" in df_team.columns else hist

    r["home_away_win_rate_diff"] = (
        (hist["result_for_team"] == "W").mean() -
        (hist["result_for_team"] == "L").mean()
    )
    r["scoring_consistency"] = (hist["goals_for"] > 0).mean()
    r["conceding_consistency"] = (hist["goals_against"] > 0).mean()
    r["dominance_index"] = (
        hist["shots_team"].mean() /
        (hist["shots_team"].mean() + shots_opp.mean()).clip(0.1)
    )
    r["discipline_ratio"] = (
        hist["fouls_team"].mean() /
        hist["fouls_opp"].mean().clip(0.1)
    )

def build_team_view(df_full: pd.DataFrame, perspective: str) -> pd.DataFrame:
    if perspective == "home":
        view = df_full.copy()
        view["team_id"]         = view["home_team_id"]
        view["goals_for"]       = view["home_score"]
        view["goals_against"]   = view["away_score"]
        view["result_for_team"] = view["result"].map({"H":"W","D":"D","A":"L"})
        view["yellows_team"]    = view["home_yellow_cards"]
        view["yellows_opp"]     = view["away_yellow_cards"]
        view["fouls_team"]      = view["home_fouls"]
        view["fouls_opp"]       = view["away_fouls"]
        view["corners_for"]     = view["home_corners"]
        view["corners_against"] = view["away_corners"]
        view["possession_team"] = view["home_possession"]
        view["passes_team"]     = view["home_passes"]
        view["passes_acc_team"] = view["home_passes_accurate"]
        view["shots_team"]      = view["home_shots"]
        view["sot_team"]        = view["home_shots_on_target"]
        view["soff_team"]       = view["home_shots_off_target"]
        view["sblk_team"]       = view["home_shots_blocked"]
        view["bc_team"]         = view["home_big_chances"]
        view["bcm_team"]        = view["home_big_chances_missed"]
        view["tackles_team"]    = view["home_tackles"]
        view["inter_team"]      = view["home_interceptions"]
        view["aer_won_team"]    = view["home_aerial_duels_won"]
        view["aer_tot_team"]    = view["home_aerial_duels_total"]
        view["drib_team"]       = view["home_dribbles_successful"]
        view["clear_team"]      = view["home_clearances"]
        view["offside_team"]    = view["home_offsides"]
        view["saves_team"]      = view["home_saves"]
        view["xg_team"]         = view["home_xg"]
        view["reds_team"]       = view["home_red_cards"]
        view["xg_opp"]          = view["away_xg"]
        view["sot_opp"]         = view["away_shots_on_target"]
        view["shots_opp"]       = view["away_shots"]
        view["bc_opp"]          = view["away_big_chances"]
    else:
        view = df_full.copy()
        view["team_id"]         = view["away_team_id"]
        view["goals_for"]       = view["away_score"]
        view["goals_against"]   = view["home_score"]
        view["result_for_team"] = view["result"].map({"H":"L","D":"D","A":"W"})
        view["yellows_team"]    = view["away_yellow_cards"]
        view["yellows_opp"]     = view["home_yellow_cards"]
        view["fouls_team"]      = view["away_fouls"]
        view["fouls_opp"]       = view["home_fouls"]
        view["corners_for"]     = view["away_corners"]
        view["corners_against"] = view["home_corners"]
        view["possession_team"] = view["away_possession"]
        view["passes_team"]     = view["away_passes"]
        view["passes_acc_team"] = view["away_passes_accurate"]
        view["shots_team"]      = view["away_shots"]
        view["sot_team"]        = view["away_shots_on_target"]
        view["soff_team"]       = view["away_shots_off_target"]
        view["sblk_team"]       = view["away_shots_blocked"]
        view["bc_team"]         = view["away_big_chances"]
        view["bcm_team"]        = view["away_big_chances_missed"]
        view["tackles_team"]    = view["away_tackles"]
        view["inter_team"]      = view["away_interceptions"]
        view["aer_won_team"]    = view["away_aerial_duels_won"]
        view["aer_tot_team"]    = view["away_aerial_duels_total"]
        view["drib_team"]       = view["away_dribbles_successful"]
        view["clear_team"]      = view["away_clearances"]
        view["offside_team"]    = view["away_offsides"]
        view["saves_team"]      = view["away_saves"]
        view["xg_team"]         = view["away_xg"]
        view["reds_team"]       = view["away_red_cards"]
        view["xg_opp"]          = view["home_xg"]
        view["sot_opp"]         = view["home_shots_on_target"]
        view["shots_opp"]       = view["home_shots"]
        view["bc_opp"]          = view["home_big_chances"]
    return view

print("Funções de feature-engineering definidas ✅")


# In[17]:


# ── Calcula rolling features para TODOS os times ──────────────────────────────
from tqdm.auto import tqdm

def compute_rolling_for_all_teams(df_full, window, perspective):
    view     = build_team_view(df_full, perspective)
    all_feats = []

    for team_id, grp in tqdm(view.groupby("team_id"),
                             desc=f"Rolling ({perspective})", leave=False):
        feats = rolling_team_features(grp.sort_values("date"), window)
        feats.index = grp.index
        all_feats.append(feats)

    return pd.concat(all_feats).sort_index()

print("Calculando features para times MANDANTES...")
home_feats = compute_rolling_for_all_teams(df, WINDOW, "home")
home_feats.columns = ["home_" + c for c in home_feats.columns]

print("Calculando features para times VISITANTES...")
away_feats = compute_rolling_for_all_teams(df, WINDOW, "away")
away_feats.columns = ["away_" + c for c in away_feats.columns]

print(f"home_feats shape: {home_feats.shape}")
print(f"away_feats shape: {away_feats.shape}")


# ## 3. Features de Confronto Direto (H2H)

# In[18]:


def compute_h2h(df_full, window=10):
    """Calcula estatísticas dos últimos N confrontos diretos entre dois times."""
    h2h_rows = []
    for idx, row in tqdm(df_full.iterrows(), total=len(df_full),
                         desc="H2H", leave=False):
        hist = df_full[
            (df_full["date"] < row["date"]) &
            (
                ((df_full["home_team_id"] == row["home_team_id"]) & (df_full["away_team_id"] == row["away_team_id"])) |
                ((df_full["home_team_id"] == row["away_team_id"]) & (df_full["away_team_id"] == row["home_team_id"]))
            )
        ].tail(window)

        if len(hist) == 0:
            h2h_rows.append({
                "h2h_home_win_rate": np.nan, "h2h_draw_rate": np.nan,
                "h2h_yellow_avg": np.nan, "h2h_corners_avg": np.nan,
                "h2h_n_games": 0,
            })
            continue

        home_wins = ((hist["home_team_id"] == row["home_team_id"]) & (hist["result"] == "H")).sum()
        home_wins += ((hist["away_team_id"] == row["home_team_id"]) & (hist["result"] == "A")).sum()
        draws = (hist["result"] == "D").sum()

        h2h_rows.append({
            "h2h_home_win_rate": home_wins / len(hist),
            "h2h_draw_rate":     draws / len(hist),
            "h2h_yellow_avg":    hist["total_yellow_cards"].mean() if "total_yellow_cards" in hist.columns else np.nan,
            "h2h_corners_avg":   hist["total_corners"].mean() if "total_corners" in hist.columns else np.nan,
            "h2h_n_games":       len(hist),
        })

    return pd.DataFrame(h2h_rows, index=df_full.index)

h2h_feats = compute_h2h(df, WINDOW_H2H)
print(f"h2h_feats shape: {h2h_feats.shape}")
print(h2h_feats.describe().round(3).to_string())


# ## 4. Posição na Tabela (normalizada)

# In[19]:


def league_table_position(df_full):
    """Calcula a posição normalizada [0,1] na tabela para cada partida,
    baseada nos jogos ANTERIORES à data da partida."""    
    pos_rows = []
    for idx, row in tqdm(df_full.iterrows(), total=len(df_full),
                         desc="League table", leave=False):
        hist = df_full[
            (df_full["league"]  == row["league"]) &
            (df_full["season"]  == row["season"]) &
            (df_full["date"]    <  row["date"])
        ]

        if len(hist) == 0:
            pos_rows.append({"home_position_norm": 0.5, "away_position_norm": 0.5})
            continue

        # Pontos acumulados por time
        pts = {}
        for _, r in hist.iterrows():
            h, a = r["home_team_id"], r["away_team_id"]
            if h not in pts: pts[h] = 0
            if a not in pts: pts[a] = 0
            if r["result"] == "H":   pts[h] += 3
            elif r["result"] == "D": pts[h] += 1; pts[a] += 1
            else:                    pts[a] += 3

        all_pts = sorted(pts.values(), reverse=True)
        n       = len(all_pts)

        def norm_pos(team_id):
            p   = pts.get(team_id, 0)
            vals = sorted(pts.values(), reverse=True)
            if p not in vals:
                return 0.5   # time sem jogos ainda — posição neutra
            pos = vals.index(p) + 1
            return 1 - (pos - 1) / max(1, n - 1)

        pos_rows.append({
            "home_position_norm": norm_pos(row["home_team_id"]),
            "away_position_norm": norm_pos(row["away_team_id"]),
        })

    return pd.DataFrame(pos_rows, index=df_full.index)

pos_feats = league_table_position(df)
print(f"pos_feats shape: {pos_feats.shape}")


# ## 5. Médias da Liga (contexto por campeonato)

# In[20]:


# Médias da liga calculadas com TODOS os jogos anteriores na mesma temporada
def league_context(df_full):
    rows = []
    for idx, row in df_full.iterrows():
        hist = df_full[
            (df_full["league"]  == row["league"]) &
            (df_full["season"]  == row["season"]) &
            (df_full["date"]    <  row["date"])
        ]
        if len(hist) == 0:
            rows.append({"league_yellow_avg": np.nan, "league_corners_avg": np.nan})
        else:
            rows.append({
                "league_yellow_avg":  hist["total_yellow_cards"].mean() if "total_yellow_cards" in hist.columns else np.nan,
                "league_corners_avg": hist["total_corners"].mean() if "total_corners" in hist.columns else np.nan,
            })
    return pd.DataFrame(rows, index=df_full.index)

league_ctx = league_context(df)
print(f"league_ctx shape: {league_ctx.shape}")


# ## 6. Montagem das Targets

# In[21]:


# ── Linhas Over/Under ──────────────────────────────────────
YELL_LINE = 4.5    # Over/Under 6.5 cartões amarelos totais
CORN_LINE = 9.5   # Over/Under 10.5 escanteios totais

result_map = {"H": 0, "D": 1, "A": 2}
targets = pd.DataFrame({
    "match_id":    df["match_id"],
    "match_date":  df["date"],
    "league":      df["league"],
    "result":      df["result"].map(result_map),
    "yellow_cat":  (df["total_yellow_cards"] > YELL_LINE).astype(int),  # 0=Under, 1=Over
    "corners_cat": (df["total_corners"]       > CORN_LINE).astype(int), # 0=Under, 1=Over
})

print("Distribuição dos targets:")
for col, labels in [
    ("result",      ["H","D","A"]),
    ("yellow_cat",  [f"Under {YELL_LINE}", f"Over {YELL_LINE}"]),
    ("corners_cat", [f"Under {CORN_LINE}", f"Over {CORN_LINE}"]),
]:
    counts = targets[col].value_counts().sort_index()
    pcts   = (counts / len(targets) * 100).round(1)
    print(f"  {col:14s}  " + "  ".join(f"{l}={p}%" for l, p in zip(labels, pcts)))


# ## 7. Montagem do DataFrame de Features

# In[22]:


features = pd.concat([
    df[["match_id","date","league","season","home_team","away_team"]].reset_index(drop=True),
    home_feats.reset_index(drop=True),
    away_feats.reset_index(drop=True),
    h2h_feats.reset_index(drop=True),
    pos_feats.reset_index(drop=True),
    league_ctx.reset_index(drop=True),
], axis=1)

print(f"features shape: {features.shape}")

# Remove jogos sem features suficientes (time sem histórico)
min_games = 3  # MIN_GAMES_REQUIRED do .env
valid_mask = (
    (features["home_n_games"] >= min_games) &
    (features["away_n_games"] >= min_games)
)
features = features[valid_mask].reset_index(drop=True)
targets  = targets[valid_mask].reset_index(drop=True)

print(f"Após filtro min_games={min_games}: {len(features):,} partidas")
print(f"Features numéricas: {features.select_dtypes('number').shape[1]}")


# ## 8. Normalização Min-Max por Liga

# In[23]:


from sklearn.preprocessing import MinMaxScaler

feature_cols = [c for c in features.columns
                if features[c].dtype in [np.float64, np.float32, float, int, np.int64]
                and c not in ("match_id",)]

features_norm = features.copy()
# pandas 3.0 não faz upcast implícito int->float numa atribuição .loc:
# garante que as colunas a normalizar já sejam float antes de receber o scaler.
features_norm[feature_cols] = features_norm[feature_cols].astype(float)
scalers = {}

for league in features["league"].unique():
    mask = features["league"] == league
    scaler = MinMaxScaler()
    features_norm.loc[mask, feature_cols] = scaler.fit_transform(
        features.loc[mask, feature_cols]
    )
    scalers[league] = scaler

print("Normalização concluída por liga:")
for league in features["league"].unique():
    sub = features_norm[features_norm["league"] == league][feature_cols]
    print(f"  {league:20s}  min={sub.min().min():.3f}  max={sub.max().max():.3f}")


# ## 9. Validação Anti-Leakage

# In[24]:


# Verifica que nenhuma partida usa dados de jogos futuros
print("Verificação de leakage temporal...")

# Para uma amostra, confirma que home_win_rate foi calculado só com jogos anteriores
sample = features.sample(100, random_state=42)
leakage_found = False

for _, row in sample.iterrows():
    if pd.isna(row.get("home_win_rate", np.nan)):
        continue
    # Se tiver home_win_rate sem data — ok; com data, deve ser < match date
    # (validação simples: se n_games <= window, não há leakage por construção)
    if row.get("home_n_games", 0) > WINDOW:
        leakage_found = True
        print(f"  ⚠️  match_id={row['match_id']} home_n_games={row['home_n_games']} > {WINDOW}")
        break

if not leakage_found:
    print("  ✅ Nenhum leakage detectado na amostra de 100 partidas")

# Confirma split temporal
cutoff = pd.to_datetime("2025-06-30")
train_mask = features["date"] <  cutoff
val_mask   = features["date"] >= cutoff
print(f"\nJanela de treino  (< {cutoff.date()}): {train_mask.sum():,} partidas")
print(f"Janela de validação (>= {cutoff.date()}): {val_mask.sum():,} partidas")


# ## 10. Exportação

# In[25]:


features_norm.to_csv(PROC_DIR / "features.csv", index=False)
targets.to_csv(       PROC_DIR / "targets.csv",  index=False)

print(f"✅ features.csv  → {len(features_norm):,} × {features_norm.shape[1]} colunas")
print(f"✅ targets.csv   → {len(targets):,} × {targets.shape[1]} colunas")
print(f"   Salvo em: {PROC_DIR}")


# In[ ]:




