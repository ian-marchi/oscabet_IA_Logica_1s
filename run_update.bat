@echo off
REM ── Wrapper para o Task Scheduler do Windows ──────────────────────────────
REM Ativa o ambiente conda 'oscabet' e roda o pipeline de atualização semanal.
REM Ajuste o caminho do conda se o seu Miniconda estiver em outro lugar.
cd /d "%~dp0"
call "%USERPROFILE%\miniconda3\condabin\conda.bat" activate oscabet
python update_weekly.py
