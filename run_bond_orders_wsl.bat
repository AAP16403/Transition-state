@echo off
REM ============================================================
REM  Precompute GFN2-xTB Wiberg bond orders for every reaction
REM  and cache them to bond_orders_cache.pkl.
REM
REM  Why: every reaction-centre / spectator / risk mask is currently
REM  derived from a BINARY covalent-radius distance cutoff, which is
REM  blind to bond ORDER. Measured against xTB, that cutoff misses
REM  8,754 reactive atoms (22.0% of reactions get a different
REM  reactive-atom set) and mis-weights ~14% of pairs by 12x in the
REM  geometry loss.
REM
REM  One-time job: ~3 min for 40k reactions on 11 workers.
REM  Reactant + product only -- TS bond orders are deliberately NOT
REM  cached (the TS is the prediction target; using it would leak).
REM
REM  Extra flags pass through, e.g.:
REM    run_bond_orders_wsl.bat --limit 500        (smoke test)
REM    run_bond_orders_wsl.bat --workers 6
REM    run_bond_orders_wsl.bat --sample-cache-path <path>
REM ============================================================

echo Building GFN2-xTB bond-order cache in WSL (Ubuntu / venv: ~/venvs/psi)...
echo   default: all reactions from samples_cache_rgd1_v4.pkl, 11 workers
echo   note: workers are pinned to 1 BLAS thread each -- without that,
echo         thread contention made this 44 h instead of ~3 min.
echo.

wsl -d Ubuntu -u lenovo -- bash -lc "source ~/venvs/psi/bin/activate && cd '/mnt/d/Transition state' && python -u build_bond_orders.py %*"

echo.
echo ============================================================
echo  Finished (exit code %ERRORLEVEL%). Result:
echo    D:\Transition state\bond_orders_cache.pkl
echo  Failures (SCF non-convergence) are recorded per reaction id
echo  inside that file under "failures" -- never silently zeroed.
echo ============================================================
pause
