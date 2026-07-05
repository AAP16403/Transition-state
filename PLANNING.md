# PSI Pipeline - Ea Error Reduction Plan

This document lays out the next development plan for the PSI transition-state
pipeline. It is intentionally focused on analysis and experiment design, not
implementation.

Current priority: reduce validation activation-energy underprediction,
especially for rare high-barrier and complex bond-rearrangement reactions,
without degrading transition-state geometry.

---

## 0. Implementation Status

This section tracks which planned changes have actually been implemented in code.
Both `psi_full_pipeline.py` and `psi_cloud_pipeline.py` are kept in sync.

### 0.1 Completed Code Changes

All Phase 1 and Phase 2 infrastructure is implemented and ready to run.

#### Config Defaults (both pipelines)

| Setting | Old Value | New Value | Status |
|---|---|---|---|
| `ea_loss_weight` | `1.0` | `2.0` | ✅ Done |
| `ea_loss_start_epoch` | _(not present)_ | `1` | ✅ Done |
| `ea_warmup_loss_weight` | _(not present)_ | `1.0` | ✅ Done |
| `ea_warmup_epochs` | `200` | `150` | ✅ Done |
| `ea_select_weight` | `0.25` | `0.5` | ✅ Done |
| `ea_head_dropout` | `0.35` | `0.15` | ✅ Done |
| `ea_head_lr` | _(not present)_ | `3e-4` | ✅ Done |
| `ea_head_weight_decay` | _(not present)_ | `1e-3` | ✅ Done |
| `ea_detach_during_warmup` | _(not present)_ | `True` | ✅ Done |
| `epochs` | `1500` | `1200` | ✅ Done |
| `patience` | `500` | `220` | ✅ Done |
| `ea_tail_weighting_enabled` | _(not present)_ | `False` | ✅ Done (Phase 2 flag) |
| `ea_tail_weight_mode` | _(not present)_ | `"piecewise"` | ✅ Done |
| `ea_tail_weight_bins` | _(not present)_ | `[80, 100, 120]` | ✅ Done |
| `ea_tail_weight_values` | _(not present)_ | `[1.0, 1.5, 2.0, 2.5]` | ✅ Done |
| `ea_tail_weight_max` | _(not present)_ | `2.5` | ✅ Done |

#### EaHead Architecture

- ✅ Xavier-uniform initialization on the final linear layer (gain=0.1) with
  zero bias. Prevents the head from producing large random predictions at the
  start of training.

#### PSI.forward

- ✅ `detach_ea_features` argument added. When `True`, `h_ts.detach()` is passed
  to the Ea head so gradients train only the head, not the EGNN backbone.

#### build_optimizer

- ✅ Separate parameter group for the Ea head with `ea_head_lr` and
  `ea_head_weight_decay`. Backbone parameters use the main `lr` and
  `weight_decay`.

#### ea_tail_sample_weights / weighted_mean

- ✅ Piecewise sample-weight function for Phase 2 high-Ea tail weighting.
  Returns uniform `1.0` weights when `ea_tail_weighting_enabled` is `False`.
- ✅ `weighted_mean` helper for weighted reduction of per-sample losses.

#### run_epoch

- ✅ Three-phase Ea loss gating:
  - `epoch < ea_loss_start_epoch`: Ea weight = 0 (geometry-only).
  - `ea_loss_start_epoch <= epoch <= ea_warmup_epochs`: Ea weight =
    `ea_warmup_loss_weight` with detached EGNN features.
  - `epoch > ea_warmup_epochs`: Ea weight = `ea_loss_weight` with full joint
    gradient flow.
- ✅ Risk-weighted Ea loss gated to joint mode only (no risk-Ea during warmup).
- ✅ Tail-weighted Ea loss (`l_ea_weighted`) used as the actual loss term.
- ✅ Extended metrics: `ea_weighted_norm`, `ea_tail_weight` tracked per epoch.

#### train_pipeline

- ✅ `build_optimizer` replaces hardcoded single-group AdamW.
- ✅ Startup prints: base LR, Ea head LR, Ea schedule, Phase 2 weights.
- ✅ Two-column Ea MAE in epoch log (`T.EaMAE` and `V.EaMAE`).
- ✅ Extended history: `train_ea_norm`, `train_ea_weighted_norm`,
  `train_ea_tail_weight`, `val_ea_weighted_norm`, `val_ea_tail_weight`, `ea_lr`.
- ✅ Config snapshot includes `list` and `tuple` types (for bin/value arrays).
- ✅ Checkpoint resume from `psi_latest.pt` (both pipelines).
- ✅ Warmup-end message: "Ea joint warmup ended. Resetting early stopping
  tracking".
- ✅ Best model label: "best val_select" instead of "best val_geom".

#### CLI Arguments (both pipelines)

- ✅ `--ea-head-lr`
- ✅ `--ea-loss-weight`
- ✅ `--ea-warmup-loss-weight`
- ✅ `--ea-loss-start-epoch`
- ✅ `--ea-warmup-epochs`
- ✅ `--ea-select-weight`
- ✅ `--ea-head-dropout`
- ✅ `--no-ea-detach-warmup`
- ✅ `--save-dir`

#### Log File Parser

- ✅ `parse_log_content` handles both `Standard Nuclear Orientation` (Q-Chem
  intermediate steps) and `Coordinates (Angstroms)` (Q-Chem converged
  geometry) table formats. Previously only the former was parsed, which worked
  by accident because the frequency job (Job 2) reprinted the converged
  geometry under a `Standard Nuclear Orientation` header.

### 0.2 Not Yet Implemented

These items are planned for future phases and have no code yet.

| Item | Phase | Notes |
|---|---|---|
| Reaction-class loss weighting | Phase 3 | Needs formed/broken bond classification in loss |
| Geometry-tail focus weighting | Phase 4 | Needs per-sample geometry-error tracking in loss |
| Outlier audit tooling | Phase 5 | Manual inspection scripts |
| Smooth percentile Ea weighting (Design 2B) | Phase 2 alt | Only if piecewise fails |
| Error-tail oversampling (Design 2C) | Phase 2 alt | Only if loss weighting fails |
| Phase 2 CLI flags for tail weighting | Phase 2 | Enable via config edit for now; CLI flags can be added when needed |

---

## 1. Current Baseline

### 1.1 Finished Run Status

The latest completed run produced fresh outputs:

- `training_history.json`
- `detailed_analysis.json`
- `psi_best.pt`
- `psi_final.pt`
- `psi_results_dashboard.html`

Important caveat: the completed run used the older Ea settings from metadata:

- `ea_loss_weight = 1.0`
- `ea_head_dropout = 0.35`
- `ea_warmup_epochs = 200`
- `ea_select_weight = 0.25`
- `epochs = 1500`
- `patience = 500`

So this run is the baseline for comparison. It does not validate the newer
Ea warm-start/dropout/LR settings yet.

### 1.2 Baseline Validation Metrics

Learned neural Ea on validation:

- MAE: `5.007 kcal/mol`
- RMSE: `8.298 kcal/mol`
- R2: `0.8983`
- Correlation: `0.9496`
- Bias: `-1.380 kcal/mol`

Physics baseline on validation:

- MAE: `15.505 kcal/mol`
- RMSE: `19.471 kcal/mol`
- R2: `0.4397`
- Correlation: `0.6636`
- Bias: `-0.584 kcal/mol`

Geometry on validation:

- Geometry MAE: `0.11284 A`
- Median geometry MAE: `0.09065 A`
- P90 geometry MAE: `0.21287 A`
- P95 geometry MAE: `0.25925 A`

Conclusion: the neural Ea model is already much better than the physics
baseline on average, but the remaining error is concentrated in a high-barrier
and complex-rearrangement tail.

---

## 2. Main Failure Mode

### 2.1 Underprediction Bias

Validation underprediction summary:

- Underpredicted cases: `986 / 1629 = 60.5%`
- Average underprediction error: `-5.276 kcal/mol`
- Average overprediction error: `+4.595 kcal/mol`
- Overall validation bias: `-1.380 kcal/mol`

This is a mild global downward bias, but it becomes severe in high-Ea regions.

### 2.2 Error by True Ea Range

| True Ea Bin | Count | MAE | Bias | Underprediction Rate |
|---|---:|---:|---:|---:|
| `0-20` | 403 | 3.153 | +0.764 | 46.2% |
| `20-40` | 435 | 4.679 | -1.071 | 61.6% |
| `40-60` | 355 | 5.300 | -1.957 | 65.6% |
| `60-80` | 310 | 5.455 | -1.691 | 64.2% |
| `80-100` | 95 | 6.889 | -3.231 | 77.9% |
| `100-120` | 27 | 17.942 | -15.724 | 81.5% |
| `120+` | 4 | 34.925 | -34.925 | 100.0% |

Interpretation:

- The model is reliable in the common low/mid barrier region.
- The model regresses rare high barriers toward the common range.
- Above roughly `80 kcal/mol`, the model becomes increasingly conservative and
  underestimates barriers.

### 2.3 Severe Underprediction Tail

Severe underprediction threshold:

- Error `<= -20 kcal/mol`

Observed:

- `30 / 1629 = 1.84%` of validation cases.

These cases dominate RMSE and are the most important target for improvement.

Worst observed validation underpredictions:

| Reaction | True Ea | Pred Ea | Error | Geometry MAE | Notes |
|---|---:|---:|---:|---:|---|
| `rxn011859` | 142.80 | 70.62 | -72.19 | 0.176 | high barrier, C/C/H/O rearrangement |
| `rxn015365` | 100.48 | 28.36 | -72.12 | 0.212 | high barrier, C-C/C-O rearrangement |
| `rxn015305` | 107.75 | 60.80 | -46.95 | 0.237 | high barrier, multi-bond change |
| `rxn012853` | 111.72 | 66.72 | -45.00 | 0.328 | C-N rich rearrangement |
| `rxn010796` | 81.11 | 36.91 | -44.19 | 0.169 | C-N/N-N rearrangement |

---

## 3. Reaction Types in the Dataset

Reaction classes below are inferred from formed/broken bonds between reactant
and product. They are practical bond-change categories, not formal named
mechanisms.

### 3.1 Broad Reaction Families

| Broad Family | Count | MAE | Bias | Underprediction Rate | Severe Under Cases |
|---|---:|---:|---:|---:|---:|
| H/proton transfer-like | 820 | 4.653 | -1.517 | 61.1% | 12 |
| Heteroatom bond rearrangement | 316 | 5.936 | -1.423 | 63.0% | 8 |
| H2-forming elimination | 170 | 4.489 | -1.468 | 62.9% | 3 |
| C-C cleavage | 153 | 4.674 | +0.462 | 50.3% | 1 |
| C-C skeletal rearrangement | 122 | 6.697 | -2.611 | 60.7% | 6 |
| Single bond cleavage/formation | 23 | 3.265 | -0.504 | 56.5% | 0 |
| C-C formation | 13 | 4.567 | -2.941 | 53.8% | 0 |
| Mixed small-molecule rearrangement | 12 | 2.962 | -0.638 | 66.7% | 0 |

Main hard families:

1. C-C skeletal rearrangement
2. Heteroatom bond rearrangement
3. High-Ea H/proton transfer outliers

### 3.2 Common Bond-Change Signatures

Most common validation signatures:

| Bond-Change Signature | Count | MAE | Bias | Underprediction Rate |
|---|---:|---:|---:|---:|
| Form C-H, break C-C + C-H | 139 | 4.010 | -1.198 | 57.6% |
| Form H-H, break 2x C-H | 79 | 4.673 | -1.078 | 64.6% |
| Form C-H, break C-H | 69 | 5.925 | -0.975 | 68.1% |
| Form H-O, break C-H | 57 | 2.604 | -0.620 | 49.1% |
| Form H-N, break C-H | 52 | 4.170 | -0.952 | 53.8% |
| Form C-O, break C-C | 44 | 3.246 | -0.023 | 52.3% |

Harder signatures:

| Bond-Change Signature | Count | MAE | Bias | Notes |
|---|---:|---:|---:|---|
| F3/B2 | 14 | 13.413 | -4.315 | rare, high variance |
| F0/B4 | 21 | 7.484 | -1.411 | multi-cleavage |
| F3/B1 | 33 | 6.976 | -2.696 | complex formation-heavy |
| F2/B2 | 101 | 6.766 | -2.395 | common enough to target |
| F2/B3 | 53 | 6.414 | -3.595 | strong underprediction |
| F2/B1 | 93 | 6.039 | -1.481 | common enough to target |

### 3.3 Risky Hetero-Bond Patterns

Risky changed bond types and validation behavior:

| Risky Bond Pattern | Count | MAE | Bias | Underprediction Rate |
|---|---:|---:|---:|---:|
| C-N + N-N | 10 | 10.077 | -8.883 | 80.0% |
| 3x C-N | 15 | 7.933 | -3.605 | 73.3% |
| N-O | 20 | 7.081 | +2.313 | 40.0% |
| C-N + N-O | 12 | 6.221 | -0.047 | 50.0% |
| C-N | 193 | 5.710 | -1.432 | 61.1% |
| 2x C-N | 66 | 5.576 | -1.217 | 59.1% |

Priority: C-N/N-N rich reactions need targeted treatment because they show both
high MAE and strong negative bias.

---

## 4. Geometry-Ea Coupling

Ea error rises sharply with validation geometry error:

| Geometry MAE Bin | Count | Ea MAE | Bias |
|---|---:|---:|---:|
| `<0.05 A` | 311 | 2.668 | -0.830 |
| `0.05-0.10 A` | 587 | 3.822 | -1.048 |
| `0.10-0.20 A` | 516 | 6.032 | -1.829 |
| `0.20-0.30 A` | 169 | 7.573 | -1.479 |
| `>=0.30 A` | 46 | 15.021 | -3.934 |

Interpretation:

- Ea performance depends heavily on the TS geometry representation.
- The high-error tail is not purely an energy-head problem.
- Any Ea-only fix should be checked for geometry degradation.
- Complex reaction classes may need both better geometry supervision and
  stronger Ea weighting.

---

## 5. Planned Improvements

The plan is split into experiment phases. Each phase should produce a saved
history, evaluation JSON, and a small summary table before moving to the next
phase.

### Phase 0 - Freeze the Baseline

Goal: preserve the current old-config result as a baseline.

Actions:

- Archive or clearly label current `training_history.json`.
- Archive or clearly label current `detailed_analysis.json`.
- Record current `psi_final.pt` metadata.
- Save the exact baseline metrics from Section 1.

Acceptance:

- A future run can be compared against this baseline without confusion.
- The report explicitly states whether it used old or new Ea settings.

### Phase 1 - Validate Ea Warm-Start Settings

Goal: test whether the new Ea schedule improves early Ea learning and reduces
final underprediction.

This phase is the only active next step. Do not add high-Ea weighting,
reaction-class weighting, new architectures, or dataset filtering during this
phase. The purpose is to isolate the effect of the warm-start schedule.

Planned settings:

- `ea_loss_start_epoch = 1`
- `ea_detach_during_warmup = True`
- `ea_warmup_epochs = 150`
- `ea_warmup_loss_weight = 1.0`
- `ea_loss_weight = 2.0`
- `ea_select_weight = 0.5`
- `ea_head_dropout = 0.15`
- `ea_head_lr = 3e-4`
- `epochs = 1200`
- `patience = 220`

Run isolation:

- Use a new output directory, not the workspace root.
- Recommended directory: `runs/phase1_warm_start`
- Keep the old root-level baseline files unchanged.
- Confirm no `psi_latest.pt` from an old run exists inside the Phase 1 output
  directory before starting.

Recommended command:

```powershell
python psi_cloud_pipeline.py train `
  --save-dir runs/phase1_warm_start `
  --epochs 1200 `
  --patience 220 `
  --ea-loss-start-epoch 1 `
  --ea-warmup-epochs 150 `
  --ea-warmup-loss-weight 1.0 `
  --ea-loss-weight 2.0 `
  --ea-select-weight 0.5 `
  --ea-head-dropout 0.15 `
  --ea-head-lr 3e-4
```

Both `psi_cloud_pipeline.py` and `psi_full_pipeline.py` are fully synced and
support all Phase 1/2 features including `--save-dir` and checkpoint resume
from `psi_latest.pt`. Use `psi_cloud_pipeline.py` for Colab runs (it reuses
`extracted_dataset.json`). Use `psi_full_pipeline.py` locally (it can
re-extract from the tarball with `--force-extract`). The all-defaults
configuration already matches the Phase 1 settings, so passing only
`--save-dir runs/phase1_warm_start` is sufficient unless overriding defaults.

Expected startup log:

```text
Learning rates: base=1.50e-04, ea_head=3.00e-04
Ea head starts at epoch 1 on detached features; full joint Ea starts after epoch 150.
```

If those two lines do not appear, stop the run and check that the updated script
is being used.

Expected behavior:

- Initial Ea MAE should drop earlier than in the baseline.
- Validation Ea should ideally beat `5.007 kcal/mol`.
- High-Ea bin bias should become less negative.

Training milestones to inspect:

| Epoch Range | What Should Happen | Concern If |
|---|---|---|
| `1-50` | Ea head begins learning immediately from detached features. | Train Ea remains near baseline mean-prediction error around `20+ kcal/mol`. |
| `50-150` | Train and validation Ea should trend down before full joint mode. | Validation Ea is flat while train Ea drops sharply. |
| `151-300` | Full joint Ea begins; early stopping resets at epoch `151`. | Geometry jumps worse after Ea gradients reach EGNN. |
| `300-700` | Should approach or beat the old run's `5.3-5.7 kcal/mol` region faster. | Ea is worse than the old run at comparable epochs. |
| `700-1200` | Look for final improvement below old best `4.964 kcal/mol`. | Only train Ea improves while validation stays near `5.0+`. |

Minimum terminal columns to monitor:

- `Train Loss`
- `Val Loss`
- `T.Geom`
- `V.Geom`
- `T.EaMAE`
- `V.EaMAE`
- `LR`

Important interpretation:

- If `T.EaMAE` drops but `V.EaMAE` does not, the Ea head is overfitting.
- If `V.Geom` worsens while `V.EaMAE` improves slightly, the change may not be
  worth keeping.
- If both `V.Geom` and `V.EaMAE` improve, Phase 1 succeeds and becomes the new
  baseline.

Acceptance targets:

- Validation Ea MAE below `4.8 kcal/mol`, or
- Validation high-Ea `80+` MAE reduced by at least `10%`, and
- No validation geometry degradation above `0.115 A`.

Secondary targets:

- Validation Ea bias less negative than `-1.380 kcal/mol`.
- Severe underprediction count below `30`.
- High-Ea `100+` predictions less compressed toward the dataset mean.
- Validation geometry P95 not worse than `0.259 A`.

Failure signals:

- Train Ea improves but validation Ea worsens.
- Geometry MAE increases meaningfully.
- High-Ea underprediction remains unchanged.

Post-run files expected in `runs/phase1_warm_start`:

- `training_history.json`
- `detailed_analysis.json`
- `psi_best.pt`
- `psi_final.pt`
- `split_diagnostics.json`
- `psi_results_dashboard.html`

Post-run metadata check:

```powershell
python -c "import torch; ck=torch.load('runs/phase1_warm_start/psi_final.pt', map_location='cpu', weights_only=False); print(ck['metadata']['config_snapshot'])"
```

Required metadata values:

- `ea_loss_start_epoch = 1`
- `ea_detach_during_warmup = True`
- `ea_warmup_epochs = 150`
- `ea_loss_weight = 2.0`
- `ea_select_weight = 0.5`
- `ea_head_dropout = 0.15`
- `ea_head_lr = 0.0003`
- `epochs = 1200`
- `patience = 220`

Post-run analysis checklist:

1. Compare final validation Ea MAE against `5.007`.
2. Compare best validation Ea MAE against old best `4.964`.
3. Compare validation bias against `-1.380`.
4. Count severe underpredictions `<= -20 kcal/mol`.
5. Rebuild the true-Ea bin table:
   - `0-20`
   - `20-40`
   - `40-60`
   - `60-80`
   - `80-100`
   - `100-120`
   - `120+`
6. Rebuild broad reaction-family metrics:
   - H/proton transfer-like
   - H2-forming elimination
   - C-C skeletal rearrangement
   - C-C cleavage
   - Heteroatom bond rearrangement
7. Rebuild geometry-bin metrics:
   - `<0.05 A`
   - `0.05-0.10 A`
   - `0.10-0.20 A`
   - `0.20-0.30 A`
   - `>=0.30 A`

Decision after Phase 1:

- If overall validation Ea MAE improves and high-Ea bias is less negative, keep
  Phase 1 as the new baseline.
- If overall validation Ea is similar but severe underpredictions drop, keep it
  as a candidate and inspect the high-Ea cases manually.
- If only train Ea improves, reject the warm-start settings or restore stronger
  regularization.
- If geometry worsens, reduce `ea_loss_weight` or delay full joint mode.

### Phase 2 - High-Ea Tail Weighting

Goal: reduce regression-to-mean for high barriers.

This phase should happen only after Phase 1 has produced a clean warm-start
baseline. The purpose is to target the high-barrier tail directly while keeping
the normal low/mid barrier region stable.

Do not combine this phase with reaction-class weighting yet. High-Ea weighting
and reaction-family weighting should be tested separately so their effects are
not confused.

#### Phase 2 Entry Requirements

Before starting Phase 2, the Phase 1 run must be available with:

- `runs/phase1_warm_start/training_history.json`
- `runs/phase1_warm_start/detailed_analysis.json`
- `runs/phase1_warm_start/psi_final.pt`
- verified metadata showing the warm-start settings

Phase 2 should compare against two baselines:

1. Old-config baseline:
   - Validation MAE: `5.007 kcal/mol`
   - Best observed validation Ea MAE: `4.964 kcal/mol`
   - Severe underpredictions: `30`
   - Validation bias: `-1.380 kcal/mol`
2. Phase 1 warm-start baseline:
   - Fill in after Phase 1 completes.

Phase 2 should not be launched if Phase 1 metadata is wrong or if Phase 1
damaged validation geometry enough that the high-Ea analysis is no longer
comparable.

#### Failure Being Targeted

The current high-Ea failure pattern is:

| True Ea Bin | Count | MAE | Bias | Underprediction Rate |
|---|---:|---:|---:|---:|
| `80-100` | 95 | 6.889 | -3.231 | 77.9% |
| `100-120` | 27 | 17.942 | -15.724 | 81.5% |
| `120+` | 4 | 34.925 | -34.925 | 100.0% |

This suggests the model is compressing rare high barriers toward the common
mid-barrier range. Phase 2 should increase the loss contribution from high-Ea
training samples so the model is less biased toward the dense center of the Ea
distribution.

#### Candidate Weighting Designs

Test only one design at a time.

##### Design 2A - Piecewise High-Ea Weights

This is the preferred first Phase 2 design because it is simple and easy to
interpret.

Concept:

| True Ea Range | Sample Weight |
|---|---:|
| `<80 kcal/mol` | `1.0` |
| `80-100 kcal/mol` | `1.5` |
| `100-120 kcal/mol` | `2.0` |
| `>=120 kcal/mol` | `2.5` |

Reasoning:

- Most data remain unchanged.
- Rare high-barrier samples get stronger gradient.
- The cap at `2.5` avoids letting a handful of outliers dominate.

Expected effect:

- Less negative high-Ea bias.
- Lower severe underprediction count.
- Possible small increase in low/mid Ea MAE.

##### Design 2B - Smooth Ea-Percentile Weighting

Use if piecewise weighting is too sharp.

Concept:

- Weight begins increasing around the 80th or 85th percentile of train Ea.
- Weight grows smoothly up to a maximum cap.
- Suggested cap: `2.5`.

Reasoning:

- Avoids hard discontinuities at `80` and `100 kcal/mol`.
- More robust if dataset composition changes.

Risk:

- Harder to interpret than fixed Ea bins.

##### Design 2C - Error-Tail Replay / Oversampling

Use only if loss weighting is not enough.

Concept:

- Oversample high-Ea train examples during training.
- Keep validation unchanged.
- Do not duplicate validation data or change split.

Reasoning:

- Gives rare high-Ea cases more update opportunities.

Risk:

- More likely to overfit high-Ea outliers.
- Can distort epoch-to-epoch comparison because one epoch no longer means one
  pass through the original training distribution.

#### Recommended First Phase 2 Run

Run Design 2A first.

Recommended output directory:

- `runs/phase2_high_ea_piecewise`

Recommended settings:

- Start from the Phase 1 warm-start settings.
- Add piecewise Ea sample weights.
- Keep `ea_head_dropout`, `ea_head_lr`, `ea_loss_weight`, and `ea_warmup_epochs`
  unchanged from Phase 1.
- Do not add reaction-class weights.
- Do not change split seed.
- Do not change target reaction count.

Proposed run label:

- `phase2_high_ea_piecewise_v1`

Expected startup metadata additions:

- `ea_tail_weighting_enabled = True`
- `ea_tail_weight_mode = "piecewise"`
- `ea_tail_weight_bins = [80, 100, 120]`
- `ea_tail_weight_values = [1.0, 1.5, 2.0, 2.5]`
- `ea_tail_weight_max = 2.5`

If these metadata fields are not present in the final checkpoint, the run is
not auditable and should not be treated as a valid Phase 2 result.

How to run Phase 2:

The tail weighting infrastructure is fully implemented in `ea_tail_sample_weights`
and `weighted_mean`, wired into `run_epoch`. It is gated behind
`ea_tail_weighting_enabled = False` by default. To activate:

1. Set `CONFIG["ea_tail_weighting_enabled"] = True` in the config dict, or
2. Add a one-line override before calling `train_pipeline`:

```python
CONFIG["ea_tail_weighting_enabled"] = True
CONFIG["save_dir"] = "runs/phase2_high_ea_piecewise"
```

The bins, values, and max are already set to the Design 2A defaults. No CLI
flags for tail weighting exist yet — enable via config edit to keep Phase 1
runs cleanly isolated. The startup log will print the tail weighting
configuration when enabled:

```text
Phase 2 high-Ea weighting: piecewise bins=[80.0, 100.0, 120.0] values=[1.0, 1.5, 2.0, 2.5] max=2.5.
```

#### Training Behavior to Watch

| Signal | Good Behavior | Bad Behavior |
|---|---|---|
| Train Ea MAE | Improves similarly to Phase 1. | Drops much faster than validation, indicating overfit. |
| Validation Ea MAE | Same or lower than Phase 1. | Worse by more than `0.2 kcal/mol`. |
| Validation bias | Less negative. | More negative or strongly positive. |
| High-Ea bins | `80+` MAE improves. | Low/mid bins improve but high-Ea bins do not. |
| Low-Ea bin | No large overprediction. | `0-20` bin bias becomes strongly positive. |
| Geometry | Similar to Phase 1. | Validation geometry MAE rises above `0.115 A`. |

#### Required Post-Run Tables

Every Phase 2 analysis must include the following tables.

##### Overall Ea Metrics

Report for train, validation, and all:

- MAE
- RMSE
- median absolute error
- P90 absolute error
- P95 absolute error
- max absolute error
- bias
- R2
- correlation

##### True-Ea Bin Metrics

Required validation bins:

| Bin |
|---|
| `0-20` |
| `20-40` |
| `40-60` |
| `60-80` |
| `80-100` |
| `100-120` |
| `120+` |

For each bin report:

- count
- MAE
- bias
- underprediction rate
- mean true Ea
- mean predicted Ea

##### Severe Error Counts

Report:

- underprediction `<= -10 kcal/mol`
- underprediction `<= -20 kcal/mol`
- underprediction `<= -30 kcal/mol`
- overprediction `>= +10 kcal/mol`
- overprediction `>= +20 kcal/mol`
- overprediction `>= +30 kcal/mol`

The main Phase 2 success metric is severe underprediction `<= -20 kcal/mol`.

##### Geometry-Controlled Ea Metrics

Reuse geometry bins:

| Geometry MAE Bin |
|---|
| `<0.05 A` |
| `0.05-0.10 A` |
| `0.10-0.20 A` |
| `0.20-0.30 A` |
| `>=0.30 A` |

Purpose:

- Check whether high-Ea weighting actually improves energy prediction, or only
  changes behavior in cases where geometry is already bad.

##### Worst-Case Audit

List top 20 validation underpredictions with:

- `rxn_id`
- true Ea
- predicted Ea
- error
- physics-baseline error
- geometry MAE
- true Ea bin
- formed/broken bond signature

Compare specifically against the previous worst cases:

- `rxn011859`
- `rxn015365`
- `rxn015305`
- `rxn012853`
- `rxn010796`
- `rxn012116`
- `rxn013945`

#### Acceptance Targets

Primary acceptance:

- Severe underprediction count `<= -20 kcal/mol` drops from `30` to `20` or
  lower, or improves by at least `25%` relative to Phase 1.

Secondary acceptance:

- Validation `80+` Ea MAE improves by at least `10%`.
- Validation `80+` bias moves closer to zero.
- Overall validation MAE does not worsen by more than `0.2 kcal/mol`.
- Validation P95 absolute error does not increase.
- Validation geometry MAE remains at or below `0.115 A`.
- Low-Ea `0-20` bias does not become strongly positive.

Preferred successful outcome:

- Overall validation MAE is at least as good as Phase 1.
- Severe underpredictions drop materially.
- High-Ea bias is less negative.
- Geometry is unchanged.

#### Failure Conditions

Reject or revise Phase 2 weighting if:

- Overall validation MAE worsens by more than `0.2 kcal/mol`.
- Low-Ea `0-20` bin becomes overpredicted by more than `+2 kcal/mol` bias.
- Severe overpredictions `>= +20 kcal/mol` increase materially.
- Validation geometry MAE exceeds `0.115 A`.
- High-Ea underprediction count does not improve.
- Improvement comes only from one or two rare outliers while the `80-100` bin
  worsens.

#### If Design 2A Fails

If piecewise weights overfit:

- Reduce max weight from `2.5` to `2.0`.
- Reduce `100-120` weight from `2.0` to `1.75`.
- Keep `80-100` at `1.25-1.5`.

If piecewise weights are too weak:

- Keep max weight `2.5`.
- Increase only `80-100` from `1.5` to `1.75`.
- Do not exceed max weight `3.0` without manual review.

If low-Ea overprediction grows:

- Add a bias check to checkpoint selection.
- Lower all tail weights by `0.25`.
- Consider smooth percentile weighting instead of hard bins.

#### Phase 2 Exit Decision

At the end of Phase 2, choose one of three outcomes:

1. Adopt high-Ea weighting as the new default.
   - Use if it improves high-Ea tail without hurting overall validation.
2. Keep high-Ea weighting as an optional experiment flag.
   - Use if it helps high-Ea but slightly hurts low/mid Ea.
3. Reject high-Ea weighting.
   - Use if it does not reduce high-Ea underprediction or damages geometry.

### Phase 3 - Reaction-Class Weighting

Goal: specifically improve complex and risky chemistry classes.

Target classes:

- C-C skeletal rearrangement
- Heteroatom bond rearrangement
- C-N/N-N changed-bond classes
- F2/B2, F2/B3, F3/B1, F3/B2 classes

Candidate strategy:

- Use existing formed/broken bond counts.
- Use risky changed bond types.
- Apply moderate class weights in Ea loss.
- Keep geometry loss stable.

Acceptance targets:

- C-C skeletal rearrangement MAE below `6.0`.
- Heteroatom rearrangement MAE below `5.5`.
- C-N/N-N group bias improves from strongly negative toward zero.
- No overall validation MAE regression above `0.2 kcal/mol`.

### Phase 4 - Geometry Tail Improvement

Goal: reduce the geometry-driven Ea error tail.

Observation:

- Cases with geometry MAE `>=0.30 A` have Ea MAE `15.021 kcal/mol`.
- Cases with geometry MAE `<0.05 A` have Ea MAE only `2.668 kcal/mol`.

Candidate strategies:

- Increase geometry weighting on active/risk pairs.
- Add targeted geometry weight for complex bond-change reactions.
- Improve checkpoint selection so geometry tail does not degrade while Ea
  improves.
- Track validation geometry P90/P95, not only mean geometry MAE.

Acceptance targets:

- Validation geometry P95 below current `0.259 A`.
- Geometry `>=0.30 A` bin count reduced below `35`.
- Ea MAE for geometry `>=0.30 A` bin reduced below `12 kcal/mol`.

### Phase 5 - Outlier Audit

Goal: determine whether the worst high-Ea cases are valid chemistry or dataset
artifacts.

Audit list:

- `rxn011859`
- `rxn015365`
- `rxn015305`
- `rxn012853`
- `rxn010796`
- `rxn012116`
- `rxn013945`
- `rxn003640`

Checks:

- Confirm reactant/product/TS atom ordering is consistent.
- Confirm TS energy is physically plausible.
- Confirm barrier definition is correct for the desired direction.
- Visualize reactant, product, true TS, and predicted TS.
- Compare bond topology from covalent-radius cutoff against expected topology.

Possible outcomes:

- Valid hard reactions: keep and target with weighting.
- Ambiguous directionality: add forward/reverse barrier handling.
- Geometry/topology artifacts: exclude or tag as low-confidence training cases.

---

## 6. Metrics Required for Every Future Run

Every run should report:

### 6.1 Core Metrics

- Train Ea MAE
- Validation Ea MAE
- Validation Ea RMSE
- Validation Ea bias
- Validation Ea R2
- Validation Ea correlation
- Train geometry MAE
- Validation geometry MAE
- Validation geometry P90/P95

### 6.2 Tail Metrics

- Severe underprediction count: error `<= -20 kcal/mol`
- Severe overprediction count: error `>= +20 kcal/mol`
- High-Ea bin metrics:
  - `80-100`
  - `100-120`
  - `120+`
- Geometry-error bin metrics:
  - `<0.05 A`
  - `0.05-0.10 A`
  - `0.10-0.20 A`
  - `0.20-0.30 A`
  - `>=0.30 A`

### 6.3 Reaction-Class Metrics

Report MAE, bias, and underprediction rate for:

- H/proton transfer-like
- H2-forming elimination
- C-C skeletal rearrangement
- C-C cleavage
- Heteroatom bond rearrangement
- C-N changed-bond classes
- C-N/N-N changed-bond classes
- F2/B2
- F2/B3
- F3/B1
- F3/B2

---

## 7. Decision Rules

### 7.1 Keep a Change

Keep a change if at least one is true:

- Validation Ea MAE improves by `>=0.2 kcal/mol`.
- High-Ea `80+` MAE improves by `>=10%`.
- Severe underpredictions drop by `>=25%`.
- C-C skeletal or heteroatom rearrangement MAE improves by `>=10%`.

And all are true:

- Validation geometry MAE does not worsen beyond `0.115 A`.
- Validation Ea bias does not become more negative.
- P95 Ea error does not increase.

### 7.2 Reject or Rework a Change

Reject or rework if:

- Overall validation MAE improves only by overfitting train Ea.
- Validation geometry worsens materially.
- High-Ea underprediction remains unchanged.
- Low-Ea predictions become badly overestimated.
- Severe overpredictions increase strongly.

---

## 8. Proposed Experiment Order

### Experiment A - New Warm-Start Baseline

Purpose: validate the already planned Ea warm-start settings.

Compare against:

- Current baseline validation MAE: `5.007`
- Current best observed validation Ea MAE in history: `4.964`
- Current severe underprediction count: `30`

Expected result:

- Earlier Ea learning.
- Slightly lower final validation Ea.
- Less high-Ea underprediction.

### Experiment B - High-Ea Weighted Loss

Purpose: reduce high-barrier regression-to-mean.

Only run after Experiment A is understood.

Expected result:

- Better `80+` Ea bins.
- Lower severe underprediction count.
- Possible small cost in low/mid Ea MAE.

### Experiment C - Reaction-Class Weighted Loss

Purpose: target C-C skeletal and heteroatom rearrangement cases.

Expected result:

- Better complex-class MAE.
- Less negative bias in C-N/N-N groups.

### Experiment D - Geometry-Tail Focus

Purpose: reduce geometry-driven Ea failures.

Expected result:

- Lower geometry P95.
- Lower Ea MAE in geometry `>=0.30 A` bin.

### Experiment E - Outlier Audit and Dataset Labeling

Purpose: decide whether the worst cases are valid examples or dataset artifacts.

Expected result:

- Clear list of valid hard reactions.
- Clear list of questionable reactions.
- Optional dataset tags for low-confidence cases.

---

## 9. Near-Term Recommendation

Run Experiment A first. Do not add high-Ea or reaction-class weighting until the
new Ea warm-start settings have a clean result.

The next run must confirm in metadata that it used:

- `ea_loss_start_epoch`
- `ea_head_lr`
- `ea_head_dropout = 0.15`
- `ea_loss_weight = 2.0`
- `ea_warmup_epochs = 150`

If Experiment A does not reduce high-Ea underprediction, move to Experiment B.
