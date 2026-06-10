# -*- coding: utf-8 -*-
"""
scheduler.py — agendador MULTIPLATAFORMA (APScheduler) do OscaBet.

Mantém um processo rodando que dispara o update_weekly.py semanalmente.
Funciona igual no Windows e no macOS/Linux. Precisa ficar rodando (ex.: deixe
um terminal aberto, ou rode junto do app). Para um agendamento que sobrevive a
reboot SEM processo rodando, use o agendador do SO (ver README: Task Scheduler /
cron / launchd).

Uso (no ambiente conda 'oscabet'):
    python scheduler.py                         # agenda (default: segunda 04:00) e fica rodando
    python scheduler.py --now                   # roda o update UMA vez agora e sai (teste)
    python scheduler.py --day sun --hour 6      # personaliza dia/hora
    python scheduler.py --no-retrain            # repassa flag ao update_weekly
"""
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
import subprocess
import argparse
from pathlib import Path

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

ROOT = Path(__file__).resolve().parent
UPDATE = ROOT / "update_weekly.py"


def run_update(extra_args):
    print("[scheduler] disparando update_weekly…", flush=True)
    subprocess.run([sys.executable, str(UPDATE), *extra_args], cwd=str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--day", default="mon", help="dia da semana: mon,tue,wed,thu,fri,sat,sun")
    ap.add_argument("--hour", type=int, default=4)
    ap.add_argument("--minute", type=int, default=0)
    ap.add_argument("--now", action="store_true", help="roda uma vez agora e sai")
    ap.add_argument("--no-retrain", action="store_true")
    ap.add_argument("--seasons", type=int, default=2)
    args = ap.parse_args()

    extra = []
    if args.no_retrain:
        extra.append("--no-retrain")
    extra += ["--seasons", str(args.seasons)]

    if args.now:
        run_update(extra)
        return

    sched = BlockingScheduler()
    sched.add_job(lambda: run_update(extra),
                  CronTrigger(day_of_week=args.day, hour=args.hour, minute=args.minute))
    print(f"[scheduler] agendado: toda {args.day} às {args.hour:02d}:{args.minute:02d}. "
          f"Ctrl+C para sair.", flush=True)
    try:
        sched.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n[scheduler] encerrado.")


if __name__ == "__main__":
    main()
