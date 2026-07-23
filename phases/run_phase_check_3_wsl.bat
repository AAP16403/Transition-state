@echo off
REM ============================================================
REM  Phase 3: is "bigger learning rate performs worse" real?
REM
REM  The direct test. Phases 1 and 2 killed both candidate
REM  mediators (logvar clamp saturation, grad-clip saturation),
REM  so this sweeps LR itself with everything else fixed.
REM
REM  3 arms x 30 epochs x 4,000 reactions.
REM  Faster than Phase 2 per arm: --skip-final-eval drops the
REM  ~18 min evaluation/dashboard/geometry-atlas tail that no
REM  phase report reads, and which peaked near the 12 GB limit.
REM  Expect ~25 min per arm, ~75 min total.
REM
REM  A failed arm no longer kills the sweep -- Phase 2 lost its
REM  last arm to an unrelated CUDA driver fault.
REM
REM  Extra flags pass through, e.g.:
REM    run_phase_check_3_wsl.bat --analyze-only   (rebuild report only)
REM    run_phase_check_3_wsl.bat --force          (re-run completed arms)
REM ============================================================

echo Running Phase 3 learning-rate sweep in WSL...
echo   - 3 arms: lr = 5e-5 / 1.5e-4 (production) / 4.5e-4
echo   - 30 epochs each, 4,000 reactions, cosine horizon pinned to 450
echo   - post-training eval skipped; completed arms skipped unless --force
echo.

wsl -d Ubuntu -u lenovo -- bash -lc "source ~/venvs/psi/bin/activate && cd '/mnt/d/Transition state/phases' && python phase_check_3.py --data-dir '/mnt/d/Transition state/RGD1_Dataset' %*"

echo.
echo ============================================================
echo  Finished (exit code %ERRORLEVEL%). Results in:
echo    D:\Transition state\phases\PHASE3_REPORT.md   (verdict + table)
echo    D:\Transition state\phases\runs\phase3_*\     (raw run outputs)
echo ============================================================
pause
