"""
True QM validation of predicted transition states with GFN2-xTB + Sella.

For each selected reaction:
  1. Run the PSI model to predict the TS geometry.
  2. Optimize that geometry to a real first-order saddle point with Sella,
     using GFN2-xTB (via tblite) gradients/Hessian.
  3. Vibrational analysis -> count imaginary frequencies (a true TS has exactly 1).
  4. RMSD(predicted TS, QM-optimized TS) -> how close the prediction was.
  5. xTB barrier (E_TS_opt - E_reactant) vs the model's predicted Ea and the
     DFT reference Ea.
  6. IRC forward + reverse to trace the path off the saddle.

Runs on Linux/WSL: `tblite` ships a prebuilt GFN2-xTB wheel and `sella` is pure
Python, so both pip-install cleanly (Windows has no xTB wheel -> use WSL).

Install (WSL venv):  pip install tblite sella
Run:                 python xtb_qm_validation.py --samples 5
                     python xtb_qm_validation.py --strays fast_gpu_irc_results/strayed_reactions.json
"""

import os
import sys
import json
import argparse
import numpy as np

# IMPORTANT: import tblite BEFORE torch. tblite's compiled C library links against
# the system libgomp (needs GOMP_5.0), but torch bundles an older libgomp; if torch
# loads first, tblite picks up torch's crippled copy and fails to import. Loading
# tblite first makes the process resolve the system libgomp, and torch's bundled
# copy coexists fine afterward.
try:
    from tblite.ase import TBLite  # GFN2-xTB with an ASE calculator (Linux pip wheel)
except ImportError as e:
    sys.exit(f"tblite unavailable ({e}). In the WSL venv run: pip install tblite")

from ase import Atoms
from ase.vibrations import Vibrations

try:
    from sella import Sella, IRC
except ImportError:
    sys.exit("sella is not installed. In the WSL venv run: pip install sella")

import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
from psi_full_pipeline import (
    CONFIG, extract_raw_data, build_reaction_samples, make_train_val_split,
    compute_normalization, ReactionDataset, DataLoader, Subset, PSI,
    clamp_steric_collisions, kabsch_align_reactant_fragments, mds_aligned,
    move_batch_to_device, write_xyz
)

EV_TO_KCAL = 23.060548  # 1 eV -> kcal/mol


def kabsch_rmsd(P, Q):
    """Minimum RMSD between two Nx3 coordinate sets after optimal superposition."""
    P = np.asarray(P, dtype=np.float64)
    Q = np.asarray(Q, dtype=np.float64)
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)
    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    P_rot = Pc @ R.T
    return float(np.sqrt(((P_rot - Qc) ** 2).sum() / len(P)))


def count_imaginary_frequencies(freqs, cm_threshold=50.0):
    """Count imaginary vibrational modes above a magnitude threshold (cm^-1).

    ASE returns frequencies as complex numbers; genuine imaginary modes carry a
    non-zero imaginary part. The threshold discards the ~6 near-zero
    translational/rotational modes that can pick up numerical imaginary noise.
    """
    freqs = np.asarray(freqs)
    imag_cm = np.abs(freqs.imag)
    strong = imag_cm[imag_cm > cm_threshold]
    largest = float(strong.max()) if strong.size else 0.0
    return int(strong.size), largest


def _select_reactions(strays_path, val_indices, samples, num_samples):
    """Return (list of (val_idx, rxn_id)) to validate.

    If a stray list from the fast scan is given, validate those (worst first);
    otherwise fall back to the first num_samples validation reactions (all if num_samples=-1).
    """
    if strays_path:
        if not os.path.exists(strays_path):
            sys.exit(f"--strays file not found: {strays_path}")
        with open(strays_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        wanted = [(s["val_idx"], s["rxn_id"]) for s in data.get("strays", [])]
        if not wanted:
            sys.exit(f"No strays listed in {strays_path}.")
        if num_samples != -1:
            wanted = wanted[:num_samples]
        print(f"Validating {len(wanted)} strayed reactions from {strays_path}")
        return wanted

    indices = list(val_indices)
    if num_samples != -1:
        indices = indices[:num_samples]
    return [(vi, samples[vi]["rxn_id"]) for vi in indices]


def run_xtb_validation(
    ckpt_path=os.path.join(BASE_DIR, "psi_best.pt"),
    base_out_dir=os.path.join(BASE_DIR, "xtb_qm_results"),
    num_samples=-1,         # -1 means all available validation reactions

    strays_path=None,       # optional strayed_reactions.json from the fast scan
    charge=0,               # molecular charge for xTB (RGD1 is neutral CHNO)
    uhf=0,                  # number of unpaired electrons (0 = singlet)
    ts_fmax=0.05,
    ts_steps=100,
    run_irc=True,
    run_vib=True,
):
    print("=" * 70)
    print(" TRUE QM VALIDATION: GFN2-xTB (tblite) TS OPTIMIZATION & IRC ")
    print(f" Checkpoint: {ckpt_path}")
    print("=" * 70)

    if not os.path.exists(ckpt_path):
        sys.exit(f"Checkpoint not found: {ckpt_path}")

    xyz_dir = os.path.join(base_out_dir, "xyz_structures")
    irc_dir = os.path.join(base_out_dir, "irc_trajectories")
    vib_dir = os.path.join(base_out_dir, "vib_tmp")
    os.makedirs(xyz_dir, exist_ok=True)
    os.makedirs(irc_dir, exist_ok=True)
    os.makedirs(vib_dir, exist_ok=True)

    def ensure_out_dirs():
        """Re-create the output dirs if something removed them mid-run.

        The dataset-load phase between the startup makedirs and the first write
        takes minutes; a `git clean` or manual cleanup in that window deletes
        xtb_qm_results and every subsequent write dies with FileNotFoundError
        (including flush_outputs, which then kills the whole run). Warn loudly
        so the disappearance is never silent, but keep the run alive.
        """
        for d in (base_out_dir, xyz_dir, irc_dir, vib_dir):
            if not os.path.isdir(d):
                print(f"  [warn] output dir vanished mid-run, recreating: {d}")
                os.makedirs(d, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1. Load checkpoint and apply its training config so the model architecture
    #    matches the saved weights (otherwise strict=False silently drops params).
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    meta = checkpoint.get("metadata", {})
    state_dict = checkpoint["model_state_dict"]

    run_config = dict(CONFIG)
    if "config_snapshot" in meta:
        run_config.update(meta["config_snapshot"])
    run_config["device"] = str(device)
    run_config["save_dir"] = BASE_DIR                      # find the sample cache
    run_config.setdefault("data_dir", os.path.join(BASE_DIR, "RGD1_Dataset"))

    # 2. Dataset + normalization (identical split to training)
    extract_raw_data(run_config)
    samples, atom_vocab, atom_types_map = build_reaction_samples(run_config)
    train_indices, val_indices, _ = make_train_val_split(samples, run_config)
    stats = compute_normalization(samples, train_indices)
    ea_mean, ea_std = stats["ea_mean"], stats["ea_std"]

    model = PSI(run_config, len(atom_vocab)).to(device)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(f"[warn] state_dict mismatch: {len(missing)} missing, {len(unexpected)} unexpected keys "
              f"(architecture may not match the checkpoint).")
    model.eval()

    eval_dataset = ReactionDataset(run_config, samples, atom_vocab, atom_types_map, stats, is_train=False)
    use_amp = run_config.get("amp", True) and device.type == "cuda"

    targets = _select_reactions(strays_path, val_indices, samples, num_samples)
    print(f"Selected {len(targets)} reactions for QM validation.\n")

    def xtb_energy_eV(symbols, positions):
        atoms = Atoms(symbols=symbols, positions=positions)
        atoms.calc = TBLite(method="GFN2-xTB", charge=charge, uhf=uhf, verbosity=0)
        return atoms.get_potential_energy(), atoms

    json_path = os.path.join(base_out_dir, "xtb_validation_summary.json")
    md_path = os.path.join(base_out_dir, "XTB_QM_VALIDATION_REPORT.md")

    def flush_outputs(results, interrupted=False):
        """Write the JSON + MD summary from whatever is done so far.

        Called after every reaction so a mid-run stop (Ctrl-C or kill) still
        leaves complete, up-to-date results on disk.
        """
        ensure_out_dirs()
        ok = [r for r in results if r["status"] == "ok"]
        n_true_ts = sum(1 for r in ok if r.get("is_true_ts"))
        n_higher_order = sum(1 for r in ok if r.get("n_imaginary", 0) > 1)
        n_minima = sum(1 for r in ok if r.get("n_imaginary", -1) == 0)
        n_conv = sum(1 for r in ok if r.get("sella_converged"))
        rmsds = [r["rmsd_pred_vs_qmTS_A"] for r in ok if "rmsd_pred_vs_qmTS_A" in r]

        summary = {
            "checkpoint": ckpt_path,
            "interrupted": interrupted,
            "num_selected": len(targets),
            "num_processed": len(results),
            "num_ok": len(ok),
            "num_failed": len(results) - len(ok),
            "num_sella_converged": n_conv,
            "num_true_ts_1imag": n_true_ts,
            "num_higher_order_saddles": n_higher_order,
            "num_minima_0imag": n_minima,
            "median_rmsd_pred_vs_qmTS_A": float(np.median(rmsds)) if rmsds else None,
            "results": results,
        }
        tmp = json_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        os.replace(tmp, json_path)  # atomic: never leaves a half-written file

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("# GFN2-xTB QM Validation Report\n\n")
            if interrupted:
                f.write("> **Interrupted run** - shows results completed before the stop.\n\n")
            f.write(f"**Checkpoint:** `{ckpt_path}`  \n")
            f.write(f"**Reactions:** {len(results)}/{len(targets)} processed, {len(ok)} completed, "
                    f"{summary['num_failed']} failed  \n")
            f.write(f"**Sella converged (to any stationary point):** {n_conv}/{len(ok)}  \n")
            f.write(f"**Confirmed true TS (1 imaginary freq):** {n_true_ts}/{len(ok)}  \n")
            f.write(f"**Higher-order saddles (>1 imag freq, often good geometries):** {n_higher_order}/{len(ok)}  \n")
            f.write(f"**Slipped to minima (0 imag freq, failed TS):** {n_minima}/{len(ok)}  \n")
            if rmsds:
                f.write(f"**Median RMSD(predicted TS, QM TS):** {np.median(rmsds):.3f} A  \n")
            f.write("\n| Reaction | Atoms | Ea_pred | Ea_true(DFT) | xTB barrier | RMSD (A) | #imag | Converged | Outcome |\n")
            f.write("|---|---|---|---|---|---|---|---|---|\n")
            for r in results:
                if r["status"] != "ok":
                    f.write(f"| `{r['rxn_id']}` | - | - | - | - | - | - | - | FAILED |\n")
                    continue
                eap = "-" if r.get("Ea_pred_model") is None else f"{r['Ea_pred_model']:.2f}"
                f.write(f"| `{r['rxn_id']}` | {r.get('n_atoms','-')} | {eap} | "
                        f"{r.get('Ea_true_DFT',0):.2f} | {r.get('xtb_barrier_kcal',0):.2f} | "
                        f"{r.get('rmsd_pred_vs_qmTS_A',0):.3f} | {r.get('n_imaginary','-')} | "
                        f"{'yes' if r.get('sella_converged') else 'no'} | "
                        f"{'True TS' if r.get('is_true_ts') else ('Higher-order' if r.get('n_imaginary', 0) > 1 else 'Minima')} |\n")
        return summary

    results = []
    interrupted = False
    for i, (val_idx, rxn_id) in enumerate(targets):
        print(f"[Validation {i+1}/{len(targets)}] Reaction {rxn_id}")
        rec = {"rxn_id": rxn_id, "status": "ok"}
        try:
            ensure_out_dirs()
            item = eval_dataset[val_idx]
            batch = torch.utils.data.default_collate([item])
            (
                DR, DI, DP, DTS, mask, geom_mask, atom_ids, atom_phys, Ea,
                de_rxn, energy_feats, risk_pair_mask,
                risk_score, risk_penalty, complexity_flag, risky_chem_flag,
            ) = move_batch_to_device(batch, device)

            with torch.no_grad():
                with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                    p_DTS, _, ea_pred_norm = model(DR, DI, DP, mask, atom_ids, atom_phys, de_rxn, energy_feats)

            n = int(mask.sum().item())
            ea_pred = float(ea_pred_norm.float().item() * ea_std + ea_mean) if ea_pred_norm is not None else None
            ea_true = float(Ea.item())

            pred_dist = p_DTS[0, :n, :n].cpu().numpy()
            pred_dist = np.maximum((pred_dist + pred_dist.T) / 2.0, 0.0)
            np.fill_diagonal(pred_dist, 0.0)

            sample_dict = samples[val_idx]
            atom_types = sample_dict["atom_types"][:n]
            c_R = np.asarray(sample_dict["c_R"][:n], dtype=np.float64)
            c_P = np.asarray(sample_dict["c_P"][:n], dtype=np.float64)
            c_I = (kabsch_align_reactant_fragments(
                c_R, c_P, atom_types, n, run_config.get("fragment_bond_scale", 1.25))[:n] + c_P[:n]) / 2.0

            pred_dist = clamp_steric_collisions(pred_dist, atom_types)
            pred_coords = mds_aligned(pred_dist, reference_coords=c_I)

            xyz_path = os.path.join(xyz_dir, f"rxn_{rxn_id}_predicted_ts.xyz")
            write_xyz(xyz_path, atom_types, pred_coords,
                      f"predicted TS | rxn {rxn_id} | Ea_pred={ea_pred} Ea_true={ea_true:.2f}")

            # --- QM: single point on the prediction ---
            ts_atoms = Atoms(symbols=list(atom_types), positions=pred_coords)
            ts_atoms.calc = TBLite(method="GFN2-xTB", charge=charge, uhf=uhf, verbosity=0)
            e_pred_geom = ts_atoms.get_potential_energy()
            print(f"  [xTB] predicted-geometry energy: {e_pred_geom:.4f} eV")

            # --- Sella saddle-point optimization (order=1 by default) ---
            print("  [Sella] optimizing to a first-order saddle...")
            opt = Sella(ts_atoms, trajectory=os.path.join(xyz_dir, f"rxn_{rxn_id}_opt.traj"),
                        internal=True)
            opt.run(fmax=ts_fmax, steps=ts_steps)
            converged = bool(opt.converged())
            steps_taken = opt.get_number_of_steps()
            e_ts_opt = ts_atoms.get_potential_energy()
            opt_coords = ts_atoms.get_positions()
            rmsd = kabsch_rmsd(pred_coords, opt_coords)
            print(f"  [Sella] converged={converged} in {steps_taken} steps | "
                  f"RMSD(pred, QM-TS)={rmsd:.3f} A")

            rec.update({
                "n_atoms": n,
                "Ea_pred_model": ea_pred,
                "Ea_true_DFT": ea_true,
                "sella_converged": converged,
                "sella_steps": steps_taken,
                "rmsd_pred_vs_qmTS_A": rmsd,
                "E_pred_geom_eV": float(e_pred_geom),
                "E_qmTS_eV": float(e_ts_opt),
            })

            # --- xTB barrier from the optimized TS ---
            e_react, _ = xtb_energy_eV(list(atom_types), c_R)
            barrier_kcal = (e_ts_opt - e_react) * EV_TO_KCAL
            rec["xtb_barrier_kcal"] = float(barrier_kcal)
            print(f"  [xTB] barrier (E_TS_opt - E_react) = {barrier_kcal:.2f} kcal/mol "
                  f"| model Ea_pred={ea_pred if ea_pred is None else round(ea_pred,2)} "
                  f"| DFT Ea_true={ea_true:.2f}")

            # --- Vibrational analysis: imaginary-frequency check ---
            if run_vib:
                print("  [xTB] vibrational analysis (imaginary-frequency check)...")
                vib_name = os.path.join(vib_dir, f"rxn_{rxn_id}")
                vib = Vibrations(ts_atoms, name=vib_name)
                vib.run()
                freqs = vib.get_frequencies()
                vib.clean()
                n_imag, largest_imag = count_imaginary_frequencies(freqs)
                is_true_ts = (n_imag == 1)
                rec.update({"n_imaginary": n_imag, "largest_imag_cm": largest_imag,
                            "is_true_ts": is_true_ts})
                print(f"  [xTB] imaginary modes: {n_imag} (largest {largest_imag:.0f} cm^-1) "
                      f"-> {'TRUE TS' if is_true_ts else 'NOT a clean TS'}")

            # --- IRC forward + reverse from the saddle ---
            if run_irc:
                print("  [Sella] IRC forward + reverse...")
                irc = IRC(ts_atoms, trajectory=os.path.join(irc_dir, f"rxn_{rxn_id}_irc.traj"))
                irc.run(fmax=ts_fmax, steps=30, direction="forward")
                irc.run(fmax=ts_fmax, steps=30, direction="reverse")
                rec["irc_traj"] = os.path.join(irc_dir, f"rxn_{rxn_id}_irc.traj")
                print("  [Sella] IRC done.")

        except KeyboardInterrupt:
            # Save whatever finished, mark this one as interrupted, and stop.
            if "rmsd_pred_vs_qmTS_A" not in rec:
                rec["status"] = "interrupted"
            print(f"\n  [interrupted] stopping after {rxn_id}; saving completed results...")
            results.append(rec)
            interrupted = True
            break
        except Exception as e:
            rec["status"] = "failed"
            rec["error"] = f"{type(e).__name__}: {e}"
            print(f"  [ERROR] validation failed for {rxn_id}: {rec['error']}")

        results.append(rec)
        flush_outputs(results, interrupted=False)   # persist after every reaction
        print()

    summary = flush_outputs(results, interrupted=interrupted)

    print("=" * 70)
    print(" QM VALIDATION INTERRUPTED " if interrupted else " QM VALIDATION COMPLETE ")
    print(f"  processed: {summary['num_processed']}/{summary['num_selected']} | "
          f"completed: {summary['num_ok']} | true TS: {summary['num_true_ts_1imag']} | "
          f"higher-order: {summary['num_higher_order_saddles']} | minima: {summary['num_minima_0imag']} | "
          f"converged: {summary['num_sella_converged']}")
    if summary["median_rmsd_pred_vs_qmTS_A"] is not None:
        print(f"  median RMSD(pred, QM-TS): {summary['median_rmsd_pred_vs_qmTS_A']:.3f} A")
    print(f"  summary JSON: {json_path}")
    print(f"  MD report:    {md_path}")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GFN2-xTB QM validation of predicted transition states.")
    parser.add_argument("--ckpt", type=str, default=os.path.join(BASE_DIR, "psi_best.pt"))
    parser.add_argument("--samples", type=int, default=-1, help="Reactions to validate (-1 for all, each ~1-5 min)")
    parser.add_argument("--strays", type=str, default=None,
                        help="strayed_reactions.json from fast_gpu_irc_validation to validate the flagged cases")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--uhf", type=int, default=0, help="unpaired electrons (0 = closed-shell singlet)")
    parser.add_argument("--no-irc", action="store_true")
    parser.add_argument("--no-vib", action="store_true")
    args = parser.parse_args()

    run_xtb_validation(
        ckpt_path=args.ckpt,
        num_samples=args.samples,
        strays_path=args.strays,
        charge=args.charge,
        uhf=args.uhf,
        run_irc=not args.no_irc,
        run_vib=not args.no_vib,
    )
