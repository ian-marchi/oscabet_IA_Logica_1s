# -*- coding: utf-8 -*-
"""
update_weekly.py — pipeline de atualização semanal do OscaBet.

Passos:
  1) Coleta INCREMENTAL de jogos novos do Sofascore (só partidas APÓS a última
     data já existente em data/raw/matches.csv, por liga). Deduplica por match_id.
  2) Re-roda o preprocessamento (agent/preprocess.py) → features.csv / targets.csv.
  3) Re-treina o ENSEMBLE (agent/train_ensemble.py).

Cross-platform. Rode no ambiente conda 'oscabet':
    conda activate oscabet
    python update_weekly.py                  # pipeline completo
    python update_weekly.py --no-retrain     # só coleta + preprocess
    python update_weekly.py --seasons 1      # nº de temporadas recentes a varrer (default 2)
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import subprocess
import time
import argparse
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SCRAPER_DIR = ROOT / "Banco de dados" / "scraper"
LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
LOGFILE = LOG_DIR / "update_weekly.log"


def log(msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with open(LOGFILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def step_scrape() -> int:
    """Coleta incremental: calcula a última data por liga a partir do CSV atual."""
    import pandas as pd
    sys.path.insert(0, str(SCRAPER_DIR))
    from config import LEAGUES, MATCHES_CSV            # noqa: E402
    from src.client import SofascoreClient             # noqa: E402
    from src.collector import collect_league           # noqa: E402
    from src.storage import Storage                    # noqa: E402

    since = {}
    if MATCHES_CSV.exists():
        df = pd.read_csv(MATCHES_CSV, usecols=["league", "date"])
        since = df.groupby("league")["date"].max().astype(str).to_dict()

    client = SofascoreClient()
    storage = Storage()
    total = 0
    for lg, lid in LEAGUES.items():
        s = since.get(lg)
        log(f"  liga {lg}: coletando partidas após {s or '(início)'}")
        try:
            total += collect_league(client, storage, lg, lid, since_date=s)
        except Exception as e:                          # noqa: BLE001
            log(f"  ERRO na liga {lg}: {e}")
    storage.finish_run(client.request_count)
    return total


def run_step(script: Path, name: str) -> bool:
    log(f"→ {name}: {script.name}")
    r = subprocess.run([sys.executable, str(script)], cwd=str(ROOT))
    ok = (r.returncode == 0)
    log(f"  {'✓' if ok else 'FALHA'} {name}" + ("" if ok else f" (exit {r.returncode})"))
    return ok


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-retrain", action="store_true", help="pula o retreino do ensemble")
    ap.add_argument("--seasons", type=int, default=2, help="temporadas recentes a varrer por liga")
    args = ap.parse_args()
    os.environ["SCRAPER_NUM_SEASONS"] = str(args.seasons)

    t0 = time.time()
    log("=" * 60)
    log("OscaBet — atualização semanal")

    # 1) Coleta
    log("1/3 Coleta incremental do Sofascore…")
    try:
        novos = step_scrape()
    except Exception as e:                              # noqa: BLE001
        log(f"FALHA na coleta: {e}")
        return 1
    log(f"   {novos} partidas novas coletadas.")
    if novos == 0:
        log("Nenhum jogo novo — encerrando (sem preprocess/treino).")
        return 0

    # 2) Preprocessamento
    log("2/3 Preprocessamento (features/targets)…")
    if not run_step(ROOT / "agent" / "preprocess.py", "preprocess"):
        return 1

    # 3) Retreino
    if args.no_retrain:
        log("3/3 Retreino pulado (--no-retrain).")
    else:
        log("3/3 Retreino do ensemble…")
        if not run_step(ROOT / "agent" / "train_ensemble.py", "train_ensemble"):
            return 1

    log(f"Pipeline concluído em {(time.time() - t0) / 60:.1f} min.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
