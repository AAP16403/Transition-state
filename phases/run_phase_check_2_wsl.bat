@echo off
REM ============================================================
REM  Phase 2 diagnostic: is grad_clip=1.0 capping learning?
REM
REM  Phase 1 found clip_rate = 100%% on 40/40 epochs, with a
REM  median gradient norm of 6.57 against a clip of 1.0 -- every
REM  step was clipped. This sweeps the threshold across 4 arms
REM  (1.0 / 5.0 / 15.0 / off) to see whether that costs accuracy.
REM
REM  4 runs x 30 epochs x 4,000 reactions, ~45 min total.
REM  Outputs go to phases\runs\phase2_*\, NOT the repo root.
REM
REM  Extra flags pass through, e.g.:
REM    run_phase_check_2_wsl.bat --analyze-only   (rebuild report only)
REM    run_phase_check_2_wsl.bat --force         (re-run completed arms)
REM ============================================================

echo Running Phase 2 grad-clip sweep in WSL...
echo   - 4 arms: grad_clip = 1.0 / 5.0 / 15.0 / inf (off)
echo   - 30 epochs each, 4,000 reactions, cosine horizon pinned to 450
echo   - completed arms are skipped unless you pass --force
echo.

wsl -d Ubuntu -u lenovo -- bash -lc "source ~/venvs/psi/bin/activate && cd '/mnt/d/Transition state/phases' && python phase_check_2.py --data-dir '/mnt/d/Transition state/RGD1_Dataset' %*"

echo.
echo ============================================================
echo  Finished (exit code %ERRORLEVEL%). Results in:
echo    D:\Transition state\phases\PHASE2_REPORT.md   (verdict + table)
echo    D:\Transition state\phases\runs\phase2_*\     (raw run outputs)
echo ============================================================
pause
