"""Precompute GFN2-xTB Wiberg bond orders for every reaction, once, and cache them.

Why this exists
---------------
Every reaction-centre / spectator / risk mask in the pipeline is currently derived
from a BINARY covalent-radius distance cutoff (`bond_set_from_distance_matrix`,
`reaction_center_atom_mask`, `reaction_risk_features`). That cutoff is blind to bond
ORDER: a C-C single bond (1.51 A) and a C=C double bond (1.33 A) both sit far inside
`1.45 * (r_C + r_C) ~ 2.2 A`, so a pure order change registers as "no bond changed".

A fitted bond-order(length) curve was tried first and rejected: calibrating it from
mode-pairing alone made it hypersensitive (it flagged 76.5% of pairs whose length
changed by a mere 0.05-0.10 A -- ordinary conformational relaxation -- as bond-order
changes). GFN2-xTB gives the Wiberg bond order from the wavefunction instead of
inferring it from a length, so there is nothing to calibrate.

At ~0.28 s per single point this is ~6 h single-core for 40k reactions x 2 geometries,
hence the process pool and the on-disk cache: pay it once, never again per run.

Only REACTANT and PRODUCT are computed. TS bond orders are deliberately NOT cached
here: the TS is the prediction target, so feeding TS-derived quantities to the model
would be label leakage.

Run (WSL venv):
    python build_bond_orders.py                    # all reactions, 10 workers
    python build_bond_orders.py --limit 500        # smoke test
    python build_bond_orders.py --workers 6

Output: bond_orders_cache.pkl
    {"meta": {...}, "orders": {rxn_id: {"n":int, "bo_R":f16[n,n], "bo_P":f16[n,n]}},
     "failures": {rxn_id: reason}}
"""

import os

# MUST precede any tblite/numpy import, in the parent AND in every spawned worker
# (spawn re-imports this module, so module scope is the right place).
# Without this each worker's BLAS grabs all cores: 6 workers x 12 threads on 12 cores
# thrashed at 2.0 s per single point versus 0.276 s measured single-process. One
# thread per worker plus N workers is the configuration that actually scales.
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import time
import pickle
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

BOHR = 1.8897259886          # Angstrom -> Bohr; tblite works in atomic units
Z = {"H": 1, "C": 6, "N": 7, "O": 8}


def _bond_orders(numbers, positions_ang, charge, uhf):
    """Wiberg bond-order matrix [n, n] from one GFN2-xTB single point."""
    from tblite.interface import Calculator
    calc = Calculator("GFN2-xTB", numbers, positions_ang * BOHR,
                      charge=charge, uhf=uhf)
    calc.set("verbosity", 0)
    res = calc.singlepoint()
    bo = np.asarray(res.get("bond-orders"))
    if bo.ndim == 3:                     # tblite returns [n, n, nspin]
        bo = bo.sum(axis=2)
    return bo.astype(np.float16)


def _one(task):
    """Worker: (rxn_id, atom_types, c_R, c_P, charge, uhf) -> result dict."""
    rxn_id, at, cR, cP, charge, uhf = task
    try:
        numbers = np.array([Z[a] for a in at], dtype=int)
        return {"rxn_id": rxn_id, "n": len(at),
                "bo_R": _bond_orders(numbers, np.asarray(cR, float), charge, uhf),
                "bo_P": _bond_orders(numbers, np.asarray(cP, float), charge, uhf)}
    except Exception as exc:
        # Recorded, never silently substituted: an SCF failure must not become a
        # zero bond-order matrix that quietly reads as "every bond broke".
        return {"rxn_id": rxn_id, "error": f"{type(exc).__name__}: {exc}"}


def main():
    ap = argparse.ArgumentParser(description="Cache GFN2-xTB Wiberg bond orders.")
    ap.add_argument("--sample-cache-path",
                    default=os.path.join(BASE_DIR, "samples_cache_rgd1_v4.pkl"))
    ap.add_argument("--out", default=os.path.join(BASE_DIR, "bond_orders_cache.pkl"))
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--charge", type=int, default=0)
    ap.add_argument("--uhf", type=int, default=0)
    ap.add_argument("--start-method", default="spawn", choices=["spawn","forkserver","fork"],
                    help="tblite is not fork-safe; spawn is the working default")
    args = ap.parse_args()

    import multiprocessing as mp

    # tblite's C extension is not fork-safe: forked workers raise
    # "tblite C extension unimportable, cannot use C-API" even though the import
    # succeeds in the parent. 'spawn' starts each worker as a fresh interpreter
    # that loads the shared library cleanly. This module is __main__-guarded, which
    # spawn requires.
    ctx = mp.get_context(args.start_method)

    print(f"loading {args.sample_cache_path} ...")
    with open(args.sample_cache_path, "rb") as fh:
        samples = pickle.load(fh)["samples"]
    if args.limit > 0:
        samples = samples[:args.limit]
    print(f"{len(samples)} reactions -> {2*len(samples)} single points "
          f"on {args.workers} workers")

    tasks = []
    for s in samples:
        n = s["n_atoms"]
        tasks.append((s["rxn_id"], list(s["atom_types"][:n]),
                      np.asarray(s["c_R"])[:n], np.asarray(s["c_P"])[:n],
                      args.charge, args.uhf))

    orders, failures = {}, {}
    t0 = time.time()
    with ctx.Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, tasks, chunksize=16), 1):
            if "error" in r:
                failures[r["rxn_id"]] = r["error"]
            else:
                orders[r["rxn_id"]] = {"n": r["n"], "bo_R": r["bo_R"], "bo_P": r["bo_P"]}
            if i % 1000 == 0 or i == len(tasks):
                el = time.time() - t0
                rate = i / max(el, 1e-9)
                print(f"  {i}/{len(tasks)}  {el/60:.1f} min elapsed  "
                      f"{rate:.1f} rxn/s  ETA {(len(tasks)-i)/max(rate,1e-9)/60:.1f} min  "
                      f"failures={len(failures)}", flush=True)

    meta = {"version": 1, "method": "GFN2-xTB", "source": "tblite",
            "quantity": "wiberg-bond-orders", "charge": args.charge, "uhf": args.uhf,
            "sample_cache": os.path.basename(args.sample_cache_path),
            "n_reactions": len(samples)}
    with open(args.out, "wb") as fh:
        pickle.dump({"meta": meta, "orders": orders, "failures": failures}, fh,
                    protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nwrote {args.out}")
    print(f"  succeeded: {len(orders)}   failed: {len(failures)}")
    print(f"  wall time: {(time.time()-t0)/60:.1f} min")
    if failures:
        print("  first few failures:")
        for k, v in list(failures.items())[:5]:
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
