#!/bin/bash
# ── Wrapper para cron / launchd (macOS e Linux) ───────────────────────────────
# Ativa o ambiente conda 'oscabet' e roda o pipeline de atualização semanal.
cd "$(dirname "$0")" || exit 1
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate oscabet
python update_weekly.py
