@echo off
REM ============================================================
REM  Launch the PSI full pipeline (training) inside the WSL venv.
REM  Double-click this file, or run it from a terminal.
REM  Any extra flags you pass are forwarded to the train command,
REM  e.g.:  run_psi_wsl.bat --epochs 5 --compile
REM ============================================================

echo Starting PSI pipeline in WSL (Ubuntu / venv: ~/venvs/psi)...
echo.

wsl -d Ubuntu -u lenovo -- bash -lc "source ~/venvs/psi/bin/activate && cd '/mnt/d/Transition state' && python psi_full_pipeline.py train --data-dir '/mnt/d/Transition state/RGD1_Dataset' --device cuda --num-workers 2 --compile %*"

echo.
echo ============================================================
echo  Run finished (exit code %ERRORLEVEL%). Outputs are in
echo  D:\Transition state  (psi_final.pt, detailed_analysis.json,
echo  psi_results_dashboard.html).
echo ============================================================
pause
