import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

import psi_full_pipeline as p


def load_history_best(history_path):
    if not os.path.exists(history_path):
        return None
    with open(history_path, "r", encoding="utf-8") as f:
        history = json.load(f)
    if not history:
        return None
    best = min(history, key=lambda row: row.get("val_select", row.get("val_geom", float("inf"))))
    return best


def stats_from_checkpoint_or_samples(metadata, samples, train_indices):
    required = (
        "aphys_mean",
        "aphys_std",
        "ea_mean",
        "ea_std",
        "de_rxn_mean",
        "de_rxn_std",
        "efeat_mean",
        "efeat_std",
    )
    if metadata and all(key in metadata for key in required):
        return {
            "aphys_mean": np.array(metadata["aphys_mean"], dtype=np.float32),
            "aphys_std": np.array(metadata["aphys_std"], dtype=np.float32),
            "ea_mean": float(metadata["ea_mean"]),
            "ea_std": float(metadata["ea_std"]),
            "de_rxn_mean": float(metadata["de_rxn_mean"]),
            "de_rxn_std": float(metadata["de_rxn_std"]),
            "efeat_mean": np.array(metadata["efeat_mean"], dtype=np.float32),
            "efeat_std": np.array(metadata["efeat_std"], dtype=np.float32),
        }
    print("Checkpoint metadata is missing normalization stats; recomputing from train split.")
    return p.compute_normalization(samples, train_indices)


def print_stats(name, records):
    if not records:
        return
    d_maes = [r["dist_MAE"] for r in records]
    ea_metrics = p.energy_metrics(records, "Ea_pred")
    phys_metrics = p.energy_metrics(records, "Ea_pred_physics")
    print(f"\n{name} ({len(records)} reactions):")
    print(
        f"  Ea MAE (neural):       {ea_metrics['MAE']:6.2f} kcal/mol"
        f"   |  R2: {ea_metrics['R2']:7.4f}   r: {ea_metrics['Pearson']:7.4f}"
    )
    print(
        f"  Ea MAE (physics):      {phys_metrics['MAE']:6.2f} kcal/mol"
        f"   |  R2: {phys_metrics['R2']:7.4f}   (baseline)"
    )
    print(
        f"  Dist MAE:            {float(np.mean(d_maes)):6.4f} A"
        f"      |  std: {float(np.std(d_maes)):6.4f} A"
    )


def run_resumed_evaluation(args):
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint_cpu = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    metadata = dict(checkpoint_cpu.get("metadata", {}))

    config = dict(p.CONFIG)
    config.update(metadata.get("config_snapshot", {}))
    config["force_extract"] = False
    if args.dataset_json is not None:
        config["dataset_json"] = args.dataset_json
    if args.save_dir is not None:
        config["save_dir"] = args.save_dir
    if args.target_reactions is not None:
        config["target_reactions"] = args.target_reactions
    if args.batch_size is not None:
        config["batch_size"] = args.batch_size
    if args.num_workers is not None:
        config["num_workers"] = args.num_workers
    config["device"] = args.device
    config["require_cuda"] = args.require_cuda
    config["amp"] = not args.no_amp

    if not os.path.exists(config["dataset_json"]):
        raise FileNotFoundError(f"Dataset JSON not found: {config['dataset_json']}")

    device = p.resolve_device(config)
    p.configure_torch_runtime(device)
    use_amp = config["amp"] and device.type == "cuda"

    print("=" * 70)
    print(" PSI RESUME: EVALUATION FROM SAVED CHECKPOINT ")
    print("=" * 70)
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset:    {config['dataset_json']}")

    history_path = args.history or os.path.join(config["save_dir"], "training_history.json")
    best_history = load_history_best(history_path)
    if best_history:
        best_value = best_history.get("val_select", best_history.get("val_geom"))
        print(f"Training history: {history_path}")
        print(f"Best recorded epoch: {best_history.get('epoch')}  value={best_value:.4f}")

    samples, atom_vocab, atom_types_map = p.build_reaction_samples(config)
    if not samples:
        raise RuntimeError("No complete reaction triplets found.")

    n_total = len(samples)
    n_val = max(1, int(n_total * config["val_split"]))
    n_train = n_total - n_val
    rng = torch.Generator().manual_seed(config["split_seed"])
    indices = torch.randperm(n_total, generator=rng).tolist()
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]
    stats = stats_from_checkpoint_or_samples(metadata, samples, train_indices)

    eval_dataset = p.ReactionDataset(config, samples, atom_vocab, atom_types_map, stats)
    loader_kwargs = {
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "pin_memory": config["pin_memory"] and device.type == "cuda",
    }
    if config["num_workers"] > 0:
        loader_kwargs["persistent_workers"] = True
    eval_loader = DataLoader(Subset(eval_dataset, list(range(n_total))), shuffle=False, **loader_kwargs)

    model = p.PSI(config, len(atom_vocab)).to(device)
    model.load_state_dict(checkpoint_cpu["model_state_dict"])
    model.eval()

    print(f"\nData split: {n_train} train, {n_val} validation")
    print("\n" + "=" * 70)
    print(" EVALUATION (geometry + learned Ea + physics baseline) ")
    print("=" * 70)

    pred_dists_map = {}
    ea_neural_map = {}
    geom_results = []
    ea_mean, ea_std = stats["ea_mean"], stats["ea_std"]
    val_rxn_ids = {samples[vi]["rxn_id"] for vi in val_indices}

    with torch.no_grad():
        for batch in eval_loader:
            (
                _dr,
                di,
                dp_in,
                dts,
                mask,
                geom_mask,
                atom_ids,
                atom_phys,
                _ea,
                de_rxn,
                energy_feats,
                _risk_pair_mask,
                _risk_score,
                _risk_penalty,
                _complexity_flag,
                _risky_chem_flag,
            ) = p.move_batch_to_device(batch, device)
            with torch.amp.autocast(device_type=device.type, enabled=use_amp):
                pred_dts, _, ea_pred_norm = model(
                    _dr, di, dp_in, mask, atom_ids, atom_phys, de_rxn, energy_feats
                )
            ea_pred_kcal = None
            if ea_pred_norm is not None:
                ea_pred_kcal = ea_pred_norm.float().cpu().numpy() * ea_std + ea_mean
            for i in range(len(batch["rxn_id"])):
                rxn_id = batch["rxn_id"][i]
                n = int(mask[i].sum().item())
                di_np = di[i, :n, :n].cpu().numpy()
                pred_np = pred_dts[i, :n, :n].cpu().numpy()
                true_np = dts[i, :n, :n].cpu().numpy()
                gm_np = geom_mask[i, :n, :n].cpu().numpy()
                diff = np.abs(pred_np - true_np)
                dist_mae = (diff * gm_np).sum().item() / max(float(gm_np.sum()), 1.0)
                split = "val" if rxn_id in val_rxn_ids else "train"
                pred_dists_map[rxn_id] = pred_np
                if ea_pred_kcal is not None:
                    ea_neural_map[rxn_id] = float(ea_pred_kcal[i])
                geom_results.append(
                    {
                        "rxn_id": rxn_id,
                        "split": split,
                        "n_atoms": n,
                        "dist_MAE": dist_mae,
                        "dist_MAE_all": diff.mean().item(),
                        "D_I": di_np.tolist(),
                        "D_pred": pred_np.tolist(),
                        "D_true": true_np.tolist(),
                        "geom_mask": gm_np.tolist(),
                        "atom_types": atom_types_map[rxn_id],
                    }
                )

    print("\n[PhysicsEa] Recovering 3D coordinates from predicted distance matrices...")
    ea_calculator = p.PhysicsEaCalculator(
        bond_scale=config["fragment_bond_scale"],
        spectator_threshold=config["spectator_threshold"],
    )
    all_coords_ts = {}
    for sample in samples:
        rxn_id = sample["rxn_id"]
        n = sample["n_atoms"]
        atom_types = sample["atom_types"]
        pred_dist = pred_dists_map[rxn_id]
        pred_dist = np.maximum((pred_dist + pred_dist.T) / 2.0, 0.0)
        np.fill_diagonal(pred_dist, 0.0)
        pred_dist = p.clamp_steric_collisions(pred_dist, atom_types)
        c_r = np.asarray(sample["c_R"][:n], dtype=np.float64)
        c_p = np.asarray(sample["c_P"][:n], dtype=np.float64)
        c_i = (
            p.kabsch_align_reactant_fragments(
                c_r, c_p, atom_types, n, config["fragment_bond_scale"]
            )[:n]
            + c_p[:n]
        ) / 2.0
        all_coords_ts[rxn_id] = p.mds_aligned(pred_dist, reference_coords=c_i)

    train_samples = [samples[i] for i in train_indices]
    train_coords = [all_coords_ts[samples[i]["rxn_id"]] for i in train_indices]
    train_x, train_y = ea_calculator.compute_features_batch(train_samples, train_coords, config)
    ea_calculator.fit(train_x, train_y)

    samples_by_id = {sample["rxn_id"]: sample for sample in samples}
    results = []
    for geom_row in geom_results:
        rxn_id = geom_row["rxn_id"]
        sample = samples_by_id[rxn_id]
        n = sample["n_atoms"]
        c_r = np.asarray(sample["c_R"][:n], dtype=np.float64)
        c_p = np.asarray(sample["c_P"][:n], dtype=np.float64)
        c_ts_pred = all_coords_ts[rxn_id]
        de_rxn = float(sample["energy_feats_raw"][1])
        ea_true = sample["Ea_raw"]
        ea_physics = ea_calculator.predict_single(
            c_r, c_ts_pred, c_p, sample["atom_types"], n, de_rxn
        )
        ea_pred = ea_neural_map.get(rxn_id, ea_physics)
        results.append(
            {
                **geom_row,
                "Ea_true": ea_true,
                "Ea_pred": ea_pred,
                "Ea_error": abs(ea_pred - ea_true),
                "Ea_pred_physics": ea_physics,
                "Ea_error_physics": abs(ea_physics - ea_true),
            }
        )

    metadata["physics_ea_coeffs"] = ea_calculator.coeffs.tolist()
    metadata.setdefault(
        "config_snapshot",
        {key: value for key, value in config.items() if isinstance(value, (int, float, str, bool))},
    )

    train_results = [row for row in results if row["split"] == "train"]
    val_results = [row for row in results if row["split"] == "val"]
    print_stats("TRAIN SET", train_results)
    print_stats("VALIDATION SET", val_results)
    print_stats("ALL DATA", results)

    print("\nReaction        Split     Ea True    Ea Pred     Ea Err   Dist MAE")
    for row in sorted(results, key=lambda item: item["rxn_id"]):
        print(
            f"{row['rxn_id']:<15} {row['split']:<6} {row['Ea_true']:10.2f} "
            f"{row['Ea_pred']:10.2f} {row['Ea_error']:10.2f} {row['dist_MAE']:10.4f}"
        )

    output_path = args.output or os.path.join(config["save_dir"], "detailed_analysis.json")
    final_model_path = args.final_model or os.path.join(config["save_dir"], "psi_final.pt")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(final_model_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, final_model_path)
    if args.update_checkpoint:
        torch.save({"model_state_dict": model.state_dict(), "metadata": metadata}, args.checkpoint)

    print(f"\nModel saved to {final_model_path}")
    print(f"Predictions saved to {output_path}")
    if args.dashboard:
        p.create_dashboard(output_path, config["save_dir"])


def parse_args():
    parser = argparse.ArgumentParser(
        description="Resume PSI from the saved best checkpoint and run only post-training evaluation."
    )
    parser.add_argument("--checkpoint", default=os.path.join(p.CONFIG["save_dir"], "psi_best.pt"))
    parser.add_argument("--dataset-json", default=None)
    parser.add_argument("--history", default=None)
    parser.add_argument("--save-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--final-model", default=None)
    parser.add_argument("--target-reactions", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=p.CONFIG["device"])
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--dashboard", action="store_true", help="Regenerate psi_results_dashboard.html.")
    parser.add_argument(
        "--update-checkpoint",
        action="store_true",
        help="Also write fitted PhysicsEa coefficients back into the input checkpoint.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_resumed_evaluation(parse_args())
