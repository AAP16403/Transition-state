@echo off
REM ============================================================
REM  Phase 1 diagnostic: is the geometry log-variance head
REM  saturating at its -7 clamp?
REM
REM  Small-scale observation run (4,000 reactions / 40 epochs,
REM  ~15 min) plus an automatic report. Trains nothing you care
REM  about -- it writes to runs\phase1\, not the repo root, so
REM  your production psi_best.pt / training_history.json are
REM  untouched.
REM
REM  Extra flags pass through, e.g.:
REM    run_phase_check_1_wsl.bat --analyze-only   (rebuild report only)
REM ============================================================

echo Running Phase 1 log-variance observation in WSL...
echo   - 4,000 reactions, 40 epochs, cosine horizon pinned to 450
echo   - slices the existing 40k sample cache (no rebuild)
echo   - outputs go to runs\phase1\, NOT the repo root
echo.

wsl -d Ubuntu -u lenovo -- bash -lc "source ~/venvs/psi/bin/activate && cd '/mnt/d/Transition state/phases' && python phase_check_1.py --data-dir '/mnt/d/Transition state/RGD1_Dataset' %*"

echo.
echo ============================================================
echo  Finished (exit code %ERRORLEVEL%). Results in:
echo    D:\Transition state\phases\PHASE1_REPORT.md   (verdict + tables)
echo    D:\Transition state\phases\runs\phase1\       (raw run outputs)
echo ============================================================
pause
