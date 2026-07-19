@echo off
REM ============================================================
REM  One-click GFN2-xTB (tblite) QM validation of predicted TSs.
REM  Runs inside the WSL venv. Results are written after EVERY
REM  reaction, so stopping mid-run (Ctrl-C or closing the window)
REM  keeps every reaction completed up to that point.
REM
REM  Extra flags pass through, e.g.:
REM    run_xtb_validation_wsl.bat --samples 10
REM    run_xtb_validation_wsl.bat --strays fast_gpu_irc_results/strayed_reactions.json
REM ============================================================

echo Starting GFN2-xTB QM validation in WSL (this is slow: ~1-5 min per reaction)...
echo Results are saved after each reaction, so you can stop anytime with Ctrl-C.
echo.

wsl -d Ubuntu -u lenovo -- bash -lc "source ~/venvs/psi/bin/activate && cd '/mnt/d/Transition state' && python xtb_qm_validation.py %*"

echo.
echo ============================================================
echo  Finished (exit code %ERRORLEVEL%). Results in:
echo    D:\Transition state\xtb_qm_results\
echo      - xtb_validation_summary.json
echo      - XTB_QM_VALIDATION_REPORT.md
echo ============================================================
pause
