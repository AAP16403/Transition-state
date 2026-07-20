@echo off
REM ============================================================
REM  Per-sector geometry failure atlas for the PSI TS model.
REM  Runs forward passes (return_debug) inside the WSL psi venv
REM  and writes a per-reaction sector-attribution report.
REM
REM  Forward-only (no training), uses the existing sample cache.
REM  Extra flags pass through, e.g.:
REM    run_geom_diagnostics_wsl.bat --split val --limit 500
REM    run_geom_diagnostics_wsl.bat --split val            (all 6000)
REM    run_geom_diagnostics_wsl.bat --ckpt psi_geom_best.pt
REM ============================================================

echo Running geometry failure-atlas diagnostics in WSL...
echo   - chirality sector needs c_TS in the cache (rebuild samples_cache_rgd1.pkl to enable)
echo   - uncertainty sector is inert until a checkpoint trained with geom_uncertainty=True
echo.

wsl -d Ubuntu -u lenovo -- bash -lc "source ~/venvs/psi/bin/activate && cd '/mnt/d/Transition state' && python geom_diagnostics.py %*"

echo.
echo ============================================================
echo  Finished (exit code %ERRORLEVEL%). Results in:
echo    D:\Transition state\geom_diagnostics\
echo      - geom_diagnostics.json
echo      - GEOM_DIAGNOSTICS_REPORT.md
echo ============================================================
pause
