"""Per-sector geometry failure atlas for the PSI transition-state model.

Runs forward passes with `return_debug=True` to expose every stage of the
geometry pipeline, then for each reaction attributes the error to the sector
that actually broke and cross-tabs it against the failure-mode buckets from
PSI_FAILURE_ANALYSIS_REPORT.md (unimolecular / N-rich / far-from-midpoint).

Sectors instrumented
  0 reaction-type   : n_atoms, #fragments, #N, TS-vs-midpoint deviation
  1 geometry head   : coarse D-MAE, non-interpolative "envelope escape", clamp saturation
                      (INAPPLICABLE under geometry_mode='coords' -- reported as n/a)
  2 seed quality    : seed D-MAE vs true TS; MDS embedding stress |dist(x_init) -
                      D_coarse| (distance mode only), degeneracy (NaN)
  3 EGNN refiner    : atom displacement ||x_ts - x_init||, gain over whatever the
                      EGNN was seeded with (coarse matrix, or the R/P midpoint)
  4 final geometry  : D-MAE vs DFT TS, under-shoot ratio, intra- vs cross-fragment error
  5 chirality       : Kabsch RMSD proper-rotation vs reflection-allowed (enantiomer flip)
  6 uncertainty     : predicted sigma vs actual per-atom error (calibration)

Run (WSL venv):  python geom_diagnostics.py --split val --limit 500
                 python geom_diagnostics.py --split val            # all 6000
Outputs:         geom_diagnostics/geom_diagnostics.json
                 geom_diagnostics/GEOM_DIAGNOSTICS_REPORT.md
"""

import os
import sys
import json
import argparse
import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from psi_full_pipeline import (
    CONFIG, extract_raw_data, build_reaction_samples, make_train_val_split,
    compute_normalization, ReactionDataset, PSI, move_batch_to_device,
    coords_from_batch,
)

# thresholds (Angstrom) tuned to the val distribution (median dist_MAE ~0.10)
FAIL_MAE = 0.15          # a reaction counts as "failing" above this final D-MAE
HI_ESCAPE_FRAC = 0.85    # envelope escape near delta_clamp => saturating
FLIP_RATIO = 0.7         # reflection RMSD < FLIP_RATIO * proper RMSD => chirality flip
CROSS_RATIO = 1.5        # cross-frag error this much > intra => orientation failure


def _kabsch_rmsd(P, Q, allow_reflection=False):
    """RMSD after optimal superposition. allow_reflection lets an enantiomer match."""
    P = np.asarray(P, float); Q = np.asarray(Q, float)
    Pc = P - P.mean(0); Qc = Q - Q.mean(0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    if allow_reflection:
        R = Vt.T @ U.T                       # best orthogonal (may reflect)
    else:
        d = np.sign(np.linalg.det(Vt.T @ U.T))
        R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T   # proper rotation only
    return float(np.sqrt(((Pc @ R.T - Qc) ** 2).sum() / len(P)))


def _n_fragments(geom_mask, n):
    par = list(range(n))
    def find(a):
        while par[a] != a:
            par[a] = par[par[a]]; a = par[a]
        return a
    for i in range(n):
        for j in range(i + 1, n):
            if geom_mask[i][j] > 0.5:
                ra, rb = find(i), find(j)
                if ra != rb:
                    par[ra] = rb
    return len({find(i) for i in range(n)})


def _off(n):
    return ~np.eye(n, dtype=bool)


def diagnose_reaction(rxn_id, n, atom_types, DR, DP, DI, DTRUE, dbg, c_TS, geom_mask,
                      delta_clamp):
    """Compute every sector's diagnostic for one reaction. Returns a flat dict."""
    off = _off(n)
    # D_coarse is absent in coordinate-native mode (geometry_mode='coords'): there
    # is no geometry head, so sectors 1 and 2 measure a stage that does not exist.
    # They are reported as None (inapplicable) rather than 0.0, which would read as
    # "this stage was perfect" and silently distort the sector histogram.
    Dc = dbg.get("D_coarse"); Dp = dbg["D_pred"]
    coords_mode = Dc is None
    xi = dbg["x_init"]; xt = dbg["x_ts"]
    r = {"rxn_id": rxn_id, "n_atoms": n, "coords_mode": coords_mode}

    # --- sector 0: reaction type -----------------------------------------
    r["n_frag"] = _n_fragments(geom_mask, n)
    r["n_N"] = int(sum(1 for a in atom_types if a == "N"))
    dev = np.abs(DTRUE - DI)[off]
    r["middev"] = float(dev.mean()); r["maxdev"] = float(dev.max())
    r["bimolecular"] = r["n_frag"] >= 2

    # --- sector 1: geometry head (coarse interpolation) ------------------
    if coords_mode:
        r["coarse_mae"] = None
        r["envelope_escape_mean"] = r["envelope_escape_max"] = None
        r["clamp_sat_frac"] = None
    else:
        r["coarse_mae"] = float(np.abs(Dc - DTRUE)[off].mean())
        lo = np.minimum(DR, DP); hi = np.maximum(DR, DP)
        escape = (np.maximum(0.0, Dc - hi) + np.maximum(0.0, lo - Dc))[off]
        r["envelope_escape_mean"] = float(escape.mean())
        r["envelope_escape_max"] = float(escape.max())
        r["clamp_sat_frac"] = float((escape > HI_ESCAPE_FRAC * delta_clamp).mean())

    # --- sector 2: seed quality ------------------------------------------
    # seed_mae is the seed's own error against the true TS and is defined in BOTH
    # modes (MDS embedding, or the Kabsch-aligned R/P midpoint) -- it is the
    # baseline the EGNN has to beat. mds_stress is embedding fidelity |dist(x_init)
    # - D_coarse|, which only exists when there is a coarse matrix to embed.
    Dseed = np.sqrt(((xi[:, None, :] - xi[None, :, :]) ** 2).sum(-1))
    r["seed_mae"] = float(np.abs(Dseed - DTRUE)[off].mean())
    r["mds_stress"] = None if coords_mode else float(np.abs(Dseed - Dc)[off].mean())
    r["seed_nan"] = bool(np.isnan(xi).any())

    # --- sector 3: EGNN refiner ------------------------------------------
    r["egnn_disp"] = float(np.sqrt(((xt - xi) ** 2).sum(-1)).mean())
    r["refined_mae"] = float(np.abs(Dp - DTRUE)[off].mean())
    # >0 => EGNN helped. Measured against whatever the EGNN was actually handed:
    # the coarse matrix in distance mode, the midpoint seed in coords mode.
    base_mae = r["seed_mae"] if coords_mode else r["coarse_mae"]
    r["egnn_gain"] = base_mae - r["refined_mae"]
    r["egnn_gain_frac"] = r["egnn_gain"] / base_mae if base_mae > 1e-6 else 0.0

    # --- sector 4: final geometry ----------------------------------------
    r["dist_mae"] = r["refined_mae"]
    pred_dev = float(np.abs(Dp - DI)[off].mean())
    true_dev = float(dev.mean())
    r["undershoot_ratio"] = pred_dev / true_dev if true_dev > 1e-6 else float("nan")
    gm = np.asarray(geom_mask)[:n, :n].astype(bool) & off
    cross = off & ~gm
    r["intra_mae"] = float(np.abs(Dp - DTRUE)[gm].mean()) if gm.any() else 0.0
    r["cross_mae"] = float(np.abs(Dp - DTRUE)[cross].mean()) if cross.any() else 0.0

    # --- sector 5: chirality / enantiomer flip ---------------------------
    if c_TS is not None:
        rp = _kabsch_rmsd(xt, c_TS, allow_reflection=False)
        rr = _kabsch_rmsd(xt, c_TS, allow_reflection=True)
        r["rmsd_proper"] = rp; r["rmsd_reflect"] = rr
        r["chirality_flip"] = bool(rr < FLIP_RATIO * rp and rp > 0.2)
    else:
        r["rmsd_proper"] = r["rmsd_reflect"] = None; r["chirality_flip"] = False

    # --- sector 6: uncertainty calibration -------------------------------
    lv = dbg["geom_logvar"]
    if lv is not None:
        sigma = np.exp(0.5 * lv[:n])
        atom_err = np.abs(Dp - DTRUE) * off
        per_atom = atom_err.sum(1) / off.sum(1)
        if sigma.std() > 1e-8 and per_atom.std() > 1e-8:
            r["unc_calib_corr"] = float(np.corrcoef(sigma, per_atom)[0, 1])
        else:
            r["unc_calib_corr"] = None
    else:
        r["unc_calib_corr"] = None

    # --- primary failing sector (only meaningful when the reaction fails) --
    r["failing"] = r["dist_mae"] > FAIL_MAE
    r["primary_sector"] = _attribute(r)
    return r


def _attribute(r):
    """Assign the sector most responsible for this reaction's geometry error."""
    if not r["failing"]:
        return "ok"
    if r["seed_nan"]:
        return "mds_degenerate"
    if r["chirality_flip"]:
        return "chirality_flip"
    if r["cross_mae"] > CROSS_RATIO * max(r["intra_mae"], 1e-6) and r["cross_mae"] > FAIL_MAE:
        return "cross_fragment_orientation"
    # geometry head: high coarse error. Distinguish "stayed interpolative"
    # (low envelope escape) from "tried but hit the clamp". Both sectors are
    # inapplicable in coordinate-native mode, where there is no geometry head.
    if r["coarse_mae"] is not None and r["coarse_mae"] > FAIL_MAE:
        if r["clamp_sat_frac"] > 0.02:
            return "delta_clamp_saturated"
        if r["envelope_escape_mean"] < 0.05:
            return "geom_head_interpolation_bound"
        return "geom_head_error"
    if r["mds_stress"] is not None and r["mds_stress"] > 0.1:
        return "mds_lossy"
    if r["egnn_gain"] < 0:
        return "egnn_hurt"
    # Coordinate-native analogue of the geom_head sectors above: with the
    # interpolation prior gone, a surviving failure means the EGNN improved on the
    # R/P midpoint seed but did not move far enough off it. Checked AFTER
    # egnn_hurt, which is the more specific diagnosis for a negative gain. Without
    # this branch every coords-mode failure would fall through to
    # "diffuse_small_errors" and the atlas would say nothing.
    if r["coarse_mae"] is None and r.get("egnn_gain_frac", 0.0) < 0.5:
        return "egnn_underpowered"
    if r["undershoot_ratio"] == r["undershoot_ratio"] and r["undershoot_ratio"] < 0.7:
        return "residual_undershoot"
    return "diffuse_small_errors"


def run(split="val", limit=-1, ckpt=None):
    ckpt = ckpt or os.path.join(BASE_DIR, "psi_best.pt")
    out_dir = os.path.join(BASE_DIR, "geom_diagnostics")
    os.makedirs(out_dir, exist_ok=True)
    if not os.path.exists(ckpt):
        sys.exit(f"Checkpoint not found: {ckpt}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(ckpt, map_location=device, weights_only=False)
    meta = checkpoint.get("metadata", {})
    run_config = dict(CONFIG)
    if "config_snapshot" in meta:
        run_config.update(meta["config_snapshot"])
    run_config["device"] = str(device)
    run_config["save_dir"] = BASE_DIR
    run_config.setdefault("data_dir", os.path.join(BASE_DIR, "RGD1_Dataset"))

    extract_raw_data(run_config)
    samples, atom_vocab, atom_types_map = build_reaction_samples(run_config)
    train_idx, val_idx, _ = make_train_val_split(samples, run_config)
    stats = compute_normalization(samples, train_idx)
    delta_clamp = float(run_config.get("delta_clamp", 3.0))

    model = PSI(run_config, len(atom_vocab)).to(device)
    missing, unexpected = model.load_state_dict(checkpoint["model_state_dict"], strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected")
    model.eval()

    coords_mode = run_config.get("geometry_mode") == "coords"
    rc_xtb = run_config.get("rc_source", "distance") == "xtb"
    print(f"  geometry_mode={run_config.get('geometry_mode')}  "
          f"rc_source={run_config.get('rc_source')}")
    ds = ReactionDataset(run_config, samples, atom_vocab, atom_types_map, stats, is_train=False)
    indices = list(val_idx if split == "val" else train_idx)
    if limit != -1:
        indices = indices[:limit]
    print(f"Diagnosing {len(indices)} {split} reactions (delta_clamp={delta_clamp})...")

    has_cts = "c_TS" in samples[indices[0]]
    if not has_cts:
        print("[notice] c_TS coordinates absent from this sample cache -> chirality "
              "sector will be reported as NOT MEASURED. Rebuild the sample cache "
              "(delete samples_cache_rgd1.pkl) to enable the enantiomer-flip check.")

    records = []
    for k, idx in enumerate(indices):
        item = ds[idx]
        batch = torch.utils.data.default_collate([item])
        (DR, DI, DP, DTS, mask, geom_mask, atom_ids, atom_phys, Ea,
         de_rxn, energy_feats, *_rest) = move_batch_to_device(batch, device)
        # Coordinate-native forward refuses to run without the R/P midpoint seed
        # (it will not silently fall back to an MDS seed), and rc_source='xtb'
        # expects the cached Wiberg reactive-atom mask. Both come off the batch the
        # dataset already built for this config.
        c_seed = coords_from_batch(batch, device)[0] if coords_mode else None
        rc_atom = (batch["rc_atom_xtb"].to(device) if rc_xtb else None)
        with torch.no_grad():
            _, _, _, _, dbg = model(
                DR, DI, DP, mask, atom_ids, atom_phys,
                de_rxn=None, energy_feats=energy_feats, return_debug=True,
                c_seed=c_seed, rc_atom=rc_atom,
            )
        n = int(mask.sum().item())
        s = samples[idx]
        at = list(s["atom_types"][:n])
        c_TS = np.asarray(s["c_TS"][:n], float) if "c_TS" in s else None
        d = {k2: (v[0, :n, :n].cpu().numpy() if v.dim() == 3 else v[0, :n].cpu().numpy())
             for k2, v in dbg.items() if v is not None}
        d.setdefault("geom_logvar", None)
        rec = diagnose_reaction(
            s["rxn_id"], n, at,
            DR[0, :n, :n].cpu().numpy(), DP[0, :n, :n].cpu().numpy(),
            DI[0, :n, :n].cpu().numpy(), DTS[0, :n, :n].cpu().numpy(),
            d, c_TS, geom_mask[0].cpu().numpy(), delta_clamp,
        )
        records.append(rec)
        if (k + 1) % 250 == 0:
            print(f"  {k+1}/{len(indices)}")

    _write(records, out_dir, split)


def _write(records, out_dir, split):
    def med(xs): return float(np.median(xs)) if xs else 0.0
    fails = [r for r in records if r["failing"]]
    from collections import Counter
    sector_counts = Counter(r["primary_sector"] for r in records)
    fail_sectors = Counter(r["primary_sector"] for r in fails)

    # cross-tab: failure bucket x primary sector
    def bucket(r):
        b = []
        b.append("unimolecular" if r["n_frag"] == 1 else "multi-fragment")
        if r["n_N"] >= 3: b.append("N-rich")
        if r["maxdev"] > 3.0: b.append("far-from-midpoint")
        return b
    xtab = {}
    for r in fails:
        for b in bucket(r):
            xtab.setdefault(b, Counter())[r["primary_sector"]] += 1

    summary = {
        "split": split, "n": len(records), "n_failing": len(fails),
        "fail_mae_threshold": FAIL_MAE,
        "median_dist_mae": med([r["dist_mae"] for r in records]),
        "median_undershoot_ratio": med([r["undershoot_ratio"] for r in records
                                        if r["undershoot_ratio"] == r["undershoot_ratio"]]),
        "n_chirality_flips": sum(1 for r in records if r["chirality_flip"]),
        "primary_sector_all": dict(sector_counts),
        "primary_sector_failing": dict(fail_sectors),
        "records": records,
    }
    with open(os.path.join(out_dir, "geom_diagnostics.json"), "w") as f:
        json.dump(summary, f, indent=2)

    md = os.path.join(out_dir, "GEOM_DIAGNOSTICS_REPORT.md")
    with open(md, "w") as f:
        f.write("# Geometry Failure Atlas\n\n")
        f.write(f"**Split:** {split}  **Reactions:** {len(records)}  "
                f"**Failing (D-MAE > {FAIL_MAE} A):** {len(fails)} "
                f"({100*len(fails)/max(len(records),1):.1f}%)\n\n")
        n_chir = sum(1 for r in records if r["rmsd_proper"] is not None)
        chir_str = (f"{summary['n_chirality_flips']} / {n_chir} measured"
                    if n_chir else "NOT MEASURED (c_TS not in cache)")
        f.write(f"**Median D-MAE:** {summary['median_dist_mae']:.4f} A  "
                f"**Median under-shoot ratio:** {summary['median_undershoot_ratio']:.3f} "
                f"(1.0 = no hedging)  **Chirality flips:** {chir_str}\n\n")
        f.write("## Where the failing reactions break (primary sector)\n\n")
        f.write("| Sector | # failing | % of failures |\n|---|---|---|\n")
        for sec, c in fail_sectors.most_common():
            f.write(f"| `{sec}` | {c} | {100*c/max(len(fails),1):.1f}% |\n")
        f.write("\n## Failure sector by reaction-type bucket\n\n")
        for b, cnt in xtab.items():
            f.write(f"**{b}** (n={sum(cnt.values())} failing): "
                    + ", ".join(f"`{s}` {c}" for s, c in cnt.most_common()) + "\n\n")
        f.write("## Worst 20 reactions (by final D-MAE)\n\n")
        # coarse / esc / mds_str are None in coordinate-native mode (no geometry
        # head, no MDS); "n/a" keeps the column honest instead of printing a zero.
        def _f(v, p):
            return "n/a" if v is None else f"{v:.{p}f}"
        f.write("| Reaction | atoms | frag | N | maxdev | D-MAE | seed | coarse | esc | mds_str | egnn_disp | undershoot | chir | sector |\n")
        f.write("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in sorted(records, key=lambda r: -r["dist_mae"])[:20]:
            f.write(f"| `{r['rxn_id']}` | {r['n_atoms']} | {r['n_frag']} | {r['n_N']} | "
                    f"{r['maxdev']:.2f} | {r['dist_mae']:.3f} | {_f(r.get('seed_mae'), 3)} | "
                    f"{_f(r['coarse_mae'], 3)} | "
                    f"{_f(r['envelope_escape_mean'], 2)} | {_f(r['mds_stress'], 3)} | {r['egnn_disp']:.2f} | "
                    f"{r['undershoot_ratio']:.2f} | {'Y' if r['chirality_flip'] else '-'} | "
                    f"`{r['primary_sector']}` |\n")
    print(f"Wrote {md}")
    print(f"  failing sectors: {dict(fail_sectors)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Per-sector geometry failure atlas.")
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()
    run(split=args.split, limit=args.limit, ckpt=args.ckpt)
