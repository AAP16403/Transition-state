@echo off
REM ============================================================
REM  Run ALL phase checks in sequence, logging everything.
REM
REM  Phase 1  log-variance observation
REM  Phase 2  grad-clip sweep
REM  Phase 3  learning-rate sweep
REM
REM  Every subprocess line is timestamped to the millisecond and
REM  teed to phases\logs\run_all_<timestamp>.log, alongside host
REM  RAM and GPU memory samples every 30s. A failing phase does
REM  not stop the sequence.
REM
REM  By default completed runs are SKIPPED, so re-running is
REM  cheap and only fills in what is missing.
REM
REM  Extra flags pass through, e.g.:
REM    run_all_phases_wsl.bat --analyze-only   (rebuild all reports, no training)
REM    run_all_phases_wsl.bat --force          (re-run every arm from scratch)
REM    run_all_phases_wsl.bat --only 2 3       (just those phases)
REM    run_all_phases_wsl.bat --sample-seconds 10   (denser memory sampling)
REM ============================================================

echo Running ALL phase checks in WSL...
echo   - phases 1, 2, 3 in sequence; a failure does not stop the rest
echo   - completed runs are skipped unless you pass --force
echo   - full timestamped log + RAM/VRAM sampling every 30s
echo.

wsl -d Ubuntu -u lenovo -- bash -lc "source ~/venvs/psi/bin/activate && cd '/mnt/d/Transition state/phases' && python -u run_all_phases.py --data-dir '/mnt/d/Transition state/RGD1_Dataset' %*"

echo.
echo ============================================================
echo  Finished (exit code %ERRORLEVEL%; non-zero = a phase failed).
echo  Results in:
echo    D:\Transition state\phases\MASTER_REPORT.md   (summary of all phases)
echo    D:\Transition state\phases\PHASE1_REPORT.md
echo    D:\Transition state\phases\PHASE2_REPORT.md
echo    D:\Transition state\phases\PHASE3_REPORT.md
echo    D:\Transition state\phases\logs\             (full timestamped logs)
echo ============================================================
pause
