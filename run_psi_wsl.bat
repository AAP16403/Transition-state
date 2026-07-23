@echo off
REM ============================================================
REM  Launch the PSI full pipeline (training) inside the WSL venv.
REM  Double-click this file, or run it from a terminal.
REM  Any extra flags you pass are forwarded to the train command,
REM  e.g.:  run_psi_wsl.bat --epochs 5 --compile
REM ============================================================

echo Starting PSI pipeline in WSL (Ubuntu / venv: ~/venvs/psi)...
echo.

REM  NOTE: --compile is NOT passed here on purpose. CONFIG["compile"] already
REM  defaults to True, and --compile/--no-compile are a mutually exclusive group,
REM  so hardcoding --compile made "run_psi_wsl.bat --no-compile" die with
REM  "argument --no-compile: not allowed with argument --compile".
REM  Behaviour is unchanged; --no-compile now actually works.

REM  Coordinate-native geometry + xTB reaction centres are now the CONFIG DEFAULTS,
REM  so no geometry/rc flags are needed here -- a bare launch trains the current
REM  production architecture:
REM    geometry_mode = coords  (EGNN seeded from the real R/P midpoint; no
REM        interpolation prior, no MDS/eigh). Resolves to samples_cache_rgd1_v4.pkl
REM        automatically.
REM    rc_source     = xtb     (GFN2-xTB Wiberg bond orders; needs
REM        bond_orders_cache.pkl, already built).
REM  To train the legacy configuration for comparison, pass:
REM    run_psi_wsl.bat --geometry-mode distance --sample-cache-path samples_cache_rgd1.pkl --rc-source distance

wsl -d Ubuntu -u lenovo -- bash -lc "source ~/venvs/psi/bin/activate && cd '/mnt/d/Transition state' && python psi_full_pipeline.py train --data-dir '/mnt/d/Transition state/RGD1_Dataset' --device cuda --num-workers 2 %*"

echo.
echo ============================================================
echo  Run finished (exit code %ERRORLEVEL%). Outputs are in
echo  D:\Transition state  (psi_final.pt, detailed_analysis.json,
echo  psi_results_dashboard.html).
echo ============================================================
pause
