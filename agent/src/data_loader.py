"""
data_loader.py — único ponto de acesso aos dados do projeto.
Cache em memória para evitar releitura dos CSVs a cada chamada.
"""
import pandas as pd
from pathlib import Path

# Detecta a raiz automaticamente: src/ → agent/ → raiz do projeto
_SRC_DIR   = Path(__file__).resolve().parent
_BASE_DIR  = _SRC_DIR.parent.parent
RAW_DIR    = _BASE_DIR / "data" / "raw"
PROC_DIR   = _BASE_DIR / "data" / "processed"
MODELS_DIR = _SRC_DIR.parent / "models"

_cache: dict = {}

def _load(key: str, path: Path, **kwargs) -> pd.DataFrame:
    if key not in _cache:
        if not path.exists():
            raise FileNotFoundError(
                f"Arquivo não encontrado: {path}\n"
                f"Execute generate_data.py e os notebooks 01-02 primeiro."
            )
        _cache[key] = pd.read_csv(path, **kwargs)
    return _cache[key]

def matches() -> pd.DataFrame:
    return _load("matches", RAW_DIR / "matches.csv", parse_dates=["date"])

def stats() -> pd.DataFrame:
    return _load("stats", RAW_DIR / "match_stats.csv")

def features() -> pd.DataFrame:
    return _load("features", PROC_DIR / "features.csv", parse_dates=["date"])

def targets() -> pd.DataFrame:
    return _load("targets", PROC_DIR / "targets.csv", parse_dates=["match_date"])

def full() -> pd.DataFrame:
    """matches + stats em um único DataFrame."""
    return matches().merge(stats(), on="match_id", how="left")

def reload():
    """Limpa o cache — chamar após atualização do banco de dados."""
    _cache.clear()
    print("Cache limpo. Próxima leitura recarrega os CSVs do disco.")
