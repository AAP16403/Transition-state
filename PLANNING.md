# PSI Pipeline — Suggested Changes (planning only, not yet applied)

This file lists improvements found while reading `psi_full_pipeline.py`, the
README, and the repo layout. Nothing here has been implemented — it is a backlog
for review. Items are grouped by severity.

---

## A. Correctness / consistency (highest value)

### A1. Eval vs. inference post-processing mismatch — affects Ea accuracy
The TS distance matrix is post-processed differently in the two code paths:

- **Training eval** (`train_pipeline`, ~lines 1499–1511): symmetrize → zero
  diagonal → `clamp_steric_collisions` → `mds_by_fragments`. **No**
  `apply_spectator_constraints`, **no** `enforce_triangle_inequality`.
- **Inference** (`predict_transition_state`, ~lines 1630–1654): same, **plus**
  `apply_spectator_constraints` and `enforce_triangle_inequality`.

Consequence: the `PhysicsEaCalculator` OLS coefficients are fit on TS coordinates
produced *without* spectator/triangle post-processing, but at prediction time the
coordinates *do* get those steps. The distribution the coefficients were fit on
differs from what they see at inference — a silent accuracy leak for Ea.

**Fix:** factor the post-processing into one shared helper, e.g.
`postprocess_pred_distance(pred_dist, D_R, D_P, atom_types, n, fragments, config)`,
and call it from both paths so eval and inference are identical.

### A2. Activation-energy reference point
In `build_reaction_samples` (~line 622):
```python
ea = (ts_e["energy"] - max(r_e["energy"], p_e["energy"])) * hartree_to_kcal
```
Ea is measured from the *higher-energy endpoint* (max of R, P), not from the
reactant. For an endothermic reaction this is the reverse barrier, not the
forward one. If the intent is the forward barrier, this should be
`ts_e["energy"] - r_e["energy"]`. **Confirm which barrier is intended** and add a
one-line comment documenting the choice either way.

---

## B. Dead code / clarity

### B1. Dead initialization in `train_pipeline`
Line ~1491 `train_X, train_y = [], []` is overwritten at ~line 1516 by
`ea_calculator.compute_features_batch(...)`. Remove the dead assignment.
(Same for `all_coords_ts = {}` pattern — that one is used, keep it.)

### B2. `build_energy_features` is now almost entirely unused
The function computes a 20-D vector (energetics + bond-angle stats), but after
the move to `PhysicsEaCalculator` only index `[1]` (`de_rxn_signed`) is ever
read (see `compute_features_batch` ~line 1041 and `predict_transition_state`
~line 1527). The angle statistics, composition counts, etc. are dead weight
computed for every one of ~5000 samples at build time.

**Options:**
- Minimal: replace the stored `energy_feats_raw` with a single scalar
  `de_rxn_signed = e_p - e_r` and update the two read sites.
- Or keep `build_energy_features` but stop calling it in the hot path.

This also removes the per-sample bond-angle graph construction
(`bond_angles_from_coords` ×2), which is the most expensive part of
`build_reaction_samples`.

---

## C. Performance

### C1. O(n²) rxn lookup in results assembly
`train_pipeline` ~line 1522:
```python
s = next(s for s in samples if s["rxn_id"] == rxn_id)
```
runs a linear scan inside a loop over all reactions → O(n²) (n ≈ 5000).
Build `samples_by_id = {s["rxn_id"]: s for s in samples}` once and index it.

### C2. `detailed_analysis.json` size
For every reaction the eval loop stores full `D_I`, `D_pred`, `D_true`, and
`geom_mask` as nested lists (~line 1479). At ~5000 reactions × 4 × 30×30 floats
this JSON can reach hundreds of MB and is slow to (de)serialize for the
dashboard. Consider: store only the upper triangle, or only the
representative/worst subset needed by the dashboard, or switch to `.npz`.

### C3. Duplicate distance-matrix builds in physics features
`compute_reorganization_energy` and `hammond_index` each rebuild `D_R/D_TS/D_P`
from coordinates with their own O(n²) loops (lines ~909 and ~951). When both run
for the same reaction (the normal path) the matrices are computed twice. Compute
once and pass them in.

---

## D. Reproducibility / robustness

### D1. Unseeded augmentation noise
`ReactionDataset.__getitem__` uses `np.random.randn(...)` (~lines 709–710) for
coordinate noise. The train/val split is seeded (`split_seed`) but the
augmentation RNG is not, so runs aren't reproducible. Seed NumPy (and consider
a per-worker seed if `num_workers > 0` is ever used).

### D2. `print_every` interacts oddly with `improved`
Minor: the epoch line prints on `improved` regardless of `print_every`, which is
fine, but early in training nearly every epoch "improves," so the log is dense
at the start. Optional: gate verbose printing.

---

## E. Documentation

### E1. README is stale (already noted in review)
`README.md` lines 8 and 45 describe a "dual-headed Transformer + GRU" with an
"Energy Head with cross-attention to predict Ea." That head was removed; Ea is
now post-hoc via `PhysicsEaCalculator` (Marcus + Hammond + BEP, 4 OLS coeffs).
Update the Overview and Architecture sections to describe:
- Geometry-only neural model (PSICore → GeometryHead → optional EGNN refiner).
- Post-hoc physics-based Ea.

### E2. Document the EGNN toggle and physics-Ea in README usage
`CONFIG["egnn_enabled"]` and the physics-Ea coefficients saved into checkpoint
metadata (`physics_ea_coeffs`) are not mentioned anywhere user-facing.

---

## Suggested order of work
1. A1 (shared post-processing) and A2 (Ea reference) — correctness.
2. B1, B2 — remove dead/duplicated work.
3. C1 — trivial speedup; C2/C3 if dataset size becomes a pain point.
4. D1 — reproducibility.
5. E1/E2 — docs.
