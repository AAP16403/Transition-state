# PSI Transition-State Model — Failure-Mode Analysis

**Date:** 2026-07-20
**Data source:** `Archives_Master/archive_run_20260719/detailed_analysis.json` (1.3 GB full eval:
34,000 train + 6,000 val reactions), cross-referenced with `xtb_qm_results/xtb_validation_summary.json`
(1,069 GFN2-xTB–validated reactions).
**Ground-truth metric:** `dist_MAE = |D_pred − D_true|` — the predicted TS distance matrix vs the **true
DFT TS** (the same D-MAE metric Choi/TSDiff report). This is *not* the xTB-relaxed-saddle RMSD.

---

## Executive summary

The model's failures are **not primarily an overfitting / regularization problem**. Coordinate-noise
augmentation and heavier weight-decay would shrink the average train→val gap, but they cannot fix the
reactions that actually fail, because **those reactions fail for structural reasons the interpolation
architecture cannot represent.**

There are **two independent failure modes**:

1. **Geometry (structural).** The `GeometryHead` predicts the TS distance matrix as an interpolation of
   reactant and product: `D_TS = α·D_R + (1−α)·D_P + δ`, with `δ` clamped to ±3.0 Å. Reactions whose true
   TS sits **far from the R/P midpoint** — asynchronous / "late" / large-rearrangement transition states —
   require a large non-interpolative correction that the head cannot produce. These fail regardless of
   regularization.
2. **Energy (orthogonal).** The Ea fat tail is a *separate* problem, uncorrelated with any geometry-type
   feature. It concentrates on low-barrier and small reactions and is driven by the energy head, not the
   geometry.

Overfitting is real (val is 2–5× worse than train on the averages) but it sets the *baseline* accuracy;
the **specific reactions that fail are selected by structure and chemistry, not by variance.**

---

## Headline numbers (val, N = 6,000)

| Metric | Train (34k) | Val (6k) | Notes |
|---|---|---|---|
| Geometry `dist_MAE` vs DFT TS (median) | 0.046 Å | **0.098 Å** | Competitive with Choi ~0.095, TSDiff single 0.137 |
| Geometry `dist_MAE` (mean) | 0.054 Å | 0.123 Å | |
| Ea MAE | 1.29 | 4.67 kcal/mol | |
| Ea median \|err\| | 1.07 | 2.85 kcal/mol | |
| Ea RMSE | 1.65 | 7.87 kcal/mol | RMSE ≫ MAE → fat tail |
| Ea R² | 0.9969 | 0.9265 | |

On the honest ground-truth metric the **median geometry is in the published SOTA range**. The problem is
the **tail**, and the tail is selected by the two failure modes below.

---

## Failure mode A — Geometry: transition states far from the R/P midpoint

### A.1 Correlation of geometry error with reaction-type features

Pearson r between `dist_MAE` and per-reaction descriptors (val, N = 6,000):

| Feature | r with `dist_MAE` | Interpretation |
|---|---|---|
| **`maxdev`** = max\|D_true − D_midpoint\| | **+0.477** | **Strongest predictor.** One badly non-interpolative pair dominates the error. |
| `middev` = mean\|D_true − D_midpoint\| | +0.418 | Overall TS asynchronicity/position. |
| `active` = # pairs moving > 0.6 Å | +0.345 | Extent of bond reorganization. |
| `nfrag` = # fragments | −0.296 | **More fragments = easier** (see A.3). |
| `n_atoms` | −0.003 | Size is essentially irrelevant. |
| `Ea_true` | +0.040 | Barrier height irrelevant to geometry error. |

`D_midpoint` here is `D_I = (D_R_aligned + D_P)/2`, the same midpoint the interpolation head starts from.
The signal is unambiguous: **geometry error is governed by how far the TS is from the reactant/product
midpoint**, not by molecule size or barrier height.

### A.2 Error scales monotonically with midpoint deviation — and the δ-clamp is a hard wall

Binned by TS-vs-midpoint deviation quartile:

| Quartile | mean `middev` | `dist_MAE` median | Ea MAE |
|---|---|---|---|
| Q1 (near-midpoint) | 0.157 Å | 0.074 Å | 4.37 |
| Q2 | 0.234 Å | 0.090 Å | 4.66 |
| Q3 | 0.309 Å | 0.113 Å | 4.32 |
| **Q4 (far-from-midpoint)** | 0.504 Å | **0.151 Å** | 5.35 |

By the maximum single-pair deviation, against the `δ = ±3.0 Å` clamp:

| Threshold | # reactions | % of val | `dist_MAE` median |
|---|---|---|---|
| all | 6,000 | 100% | 0.098 Å |
| `maxdev` > 2.0 Å | 1,228 | 20.5% | 0.165 Å |
| `maxdev` > 2.5 Å | 572 | 9.5% | 0.182 Å |
| **`maxdev` > 3.0 Å (≥ clamp)** | 231 | **3.9%** | **0.194 Å** |
| `maxdev` > 3.5 Å | 93 | 1.6% | 0.196 Å |

~**4% of reactions require a correction at or beyond the ±3.0 Å clamp** and cannot be represented at all;
their error plateaus (~0.19 Å) because the head saturates. This is an **architectural ceiling**, not a
training-variance issue — no amount of coordinate noise moves it.

### A.3 Bimolecularity inverts the naive expectation

| # fragments | N | `dist_MAE` median | Ea MAE |
|---|---|---|---|
| 1 (unimolecular) | 3,816 | **0.117 Å** | 4.76 |
| 2 | 1,462 | 0.085 Å | 5.10 |
| 3 | 720 | **0.053 Å** | 3.35 |

**More fragments = better geometry.** A bi/tri-molecular reaction is two/three largely rigid fragments
approaching; most intra-fragment distances barely change and are trivial to predict, so only the
forming-bond region is hard. A **unimolecular rearrangement** (ring closure, isomerization, concerted
shift) reorganizes the *whole* molecule — the entire matrix moves, the TS is far from any R/P midpoint,
and the interpolation prior is weakest. **Intramolecular rearrangements are the hard structural class.**
(This coexists with the earlier observation that cross-fragment pairs are individually 1.26× harder than
intra-fragment ones: those pairs are hard but few, so multi-fragment reactions still win on average.)

### A.4 Nitrogen chemistry is a distinct hard type

| Subset | N | `dist_MAE` median |
|---|---|---|
| 0 nitrogen atoms | 774 | 0.090 Å |
| ≥ 3 nitrogen atoms | 1,544 | 0.110 Å (+22%) |

N-fraction of heavy atoms: best-geometry decile 0.228 → worst-geometry decile 0.264. Nitrogen's variable
valence, lone pairs, and N–N / N–O bonding give more complex, less interpolation-friendly transition
states. C, H, and O show no such enrichment (ratios ~1.0).

---

## Failure mode B — Energy: a separate, orthogonal problem

The Ea fat tail (val RMSE 7.87 ≫ MAE 4.67; 5.3% of reactions with \|err\| > 15 kcal/mol) is **not**
explained by any geometry-type feature:

| Feature | r with Ea \|error\| |
|---|---|
| `maxdev` | +0.069 |
| `middev` | +0.065 |
| `active` | +0.005 |
| `n_atoms` | −0.082 |
| `Ea_true` | −0.045 |

All \|r\| < 0.1. Geometry-fail and Ea-fail reactions overlap only ~23%. Where the Ea error *does*
concentrate (from the binned analysis):

- **Low barriers** (Ea_true < 20 kcal/mol): MAE 7.96 vs ~4.6 for the bulk.
- **Small molecules** (5–9 atoms): MAE 6.99 — too little context for the pooling heads.
- Signed error is symmetric (51% over- / 49% under-predict, near-zero bias) → fat tails, not a systematic offset.

**Hypothesis:** the Ea head leans on its Bell-Evans-Polanyi / reaction-energy prior; kinetically
"surprising" reactions (a low true barrier where the trend predicts high, or vice versa) get pulled toward
the trend and overshoot. This is an energy-head problem to be fixed independently of geometry.

---

## The "always-fail" set (fails in both eval geometry AND xTB frequency)

23 reactions have eval `dist_MAE` > 0.2 Å **and** are not a clean TS at the xTB level. Representative cases:

| Reaction | atoms | N | `middev` | `maxdev` | eval `dist_MAE` | xTB #imag | xTB RMSD |
|---|---|---|---|---|---|---|---|
| MR_235551_0 | 20 | 2 | 0.59 | 2.94 | 0.641 | 2 | 0.25 |
| MR_982_1 | 15 | 0 | 1.07 | 3.98 | 0.347 | 2 | 0.93 |
| MR_516129_1 | 17 | 2 | 0.65 | 2.51 | 0.345 | 2 | 0.45 |
| MR_502889_0 | 15 | 1 | 0.34 | 2.25 | 0.338 | 3 | 0.81 |
| MR_167302_1 | 20 | 5 | 0.42 | 2.29 | 0.313 | 2 | 0.70 |
| MR_487643_0 | 22 | 3 | 0.44 | 2.01 | 0.309 | 0 | 0.34 |
| MR_500639_0 | 24 | 1 | 0.46 | 2.44 | 0.276 | 0 | 1.28 |

Most land near a saddle but as **higher-order saddles** (2–3 imaginary modes) — the model gets the region
but not the precise curvature. They cluster on high `maxdev` and/or high nitrogen count, consistent with
mode A.

**Important masking effect:** the xTB true-TS rate is only mildly lower for high-`maxdev` reactions
(86% for maxdev > 2.5 Å vs 90% for ≤ 2.5 Å), because **xTB re-optimization does the work the model could
not** — it relaxes a poor prediction into a nearby saddle. So the structural failure is far more visible in
the direct `dist_MAE` (mode A) than in the xTB metric. Judging the model by the xTB rate alone
under-reports this failure mode.

---

## Why coordinate noise / overfitting is not the primary lever

- The failing reactions are selected by **`maxdev`** (r ≈ 0.48), a property of the *reaction*, not of
  training variance. A reaction whose TS the interpolation head structurally cannot reach fails on both
  train and val.
- **4% of reactions saturate the δ = ±3.0 Å clamp** — a hard architectural ceiling that regularization
  cannot touch.
- Coordinate noise reduces variance (helps the average, and the train→val gap), but it does not add
  representational capacity for non-interpolative transition states, nor does it help the (orthogonal)
  energy tail.

Overfitting should still be addressed — it lifts the whole curve — but it is a **second-order** lever
relative to the structural ceiling.

---

## Recommendations (in priority order)

1. **Relax the interpolation bottleneck (mode A, highest leverage).**
   - Raise or remove the `δ` clamp (`delta_clamp`, currently 3.0 Å); it is actively saturating on ~4% of
     reactions. Consider a soft/learned bound instead of a hard clamp.
   - Give the geometry head a path to predict *non-interpolative* TS geometry directly (e.g. let the EGNN
     refiner contribute larger displacements, or add a residual branch not tied to the R/P segment).
2. **Target the hard chemistry.** Up-weight unimolecular rearrangements and nitrogen-rich reactions in the
   loss, or oversample them — they are systematically under-served.
3. **Fix the energy tail separately (mode B).** Probe the BEP-term dominance on the 57 Ea outliers; add a
   low-barrier / small-molecule–focused Ea loss term; the geometry head should not be touched for this.
4. **Then address overfitting** (coord-noise augmentation, heavier dropout/weight-decay) to lift the
   average and close the 2–5× train→val gap.
5. **Report geometry with `dist_MAE` vs DFT TS, not the xTB-relaxed RMSD** — the xTB metric masks exactly
   the structural failures this analysis identifies.

---

## Fixes applied (2026-07-20)

Implemented in `psi_full_pipeline.py`, verified by static compile + a synthetic forward/backward unit
test. **Not yet trained** (training runs elsewhere) — a short smoke run is recommended before a full run.
Not mirrored to `psi_cloud`/`psi_swarm` (older divergent loss: they scale Huber inputs and lack the
role-weighting; they also still carry the old `PlateauWarmupScheduler` and padded `torch_mds_coords`).

This section merges the earlier standalone `failure.md`, which added two further root causes on top of the
under-shoot diagnosis below: (1) **LR-schedule under-utilization** — the plateau scheduler warmed up for 40
epochs then sat at peak LR for 100+ epochs before its first decay, wasting compute and overshooting minima
(→ cosine schedule); and (2) **geometry-MAE saturation** — validation geometry MAE floored (~0.16 Å) while
train error kept dropping, because the MDS `eigh` gradient was severed by padding, starving the GeometryHead
of 3D-refinement feedback (→ per-molecule unpadded differentiable MDS). Both are in the table below.

**A diagnostic that reframed the fix.** Before changing anything, a check measured `|D_pred − midpoint|` vs
`|D_true − midpoint|`: the model **systematically under-shoots** — `pred_dev = 0.86 × true_dev`, under-shoot
in 85–94% of reactions across *all* quartiles, and on the hardest cases (true maxdev > 3.0 Å) it reaches
only **66% of the needed displacement**. Critically it under-shoots *even inside* the ±3.0 clamp, so the
clamp is not the main cause — the **loss rewards hedging toward the midpoint**. That moved the fix from
"architecture" to "loss + uncertainty".

| Change | Mechanism | Config | Status |
|---|---|---|---|
| Soft δ-clamp | `delta = dc·tanh(delta/dc)`, raised 3.0→5.0; hard clamp zeroed the gradient past the bound | `delta_clamp` | ✅ verified |
| Movement-aware weighting | Lift each pair's loss weight toward the active weight ∝ its R→P movement `|D_R−D_P|`, so moving "spectator" pairs (unimolecular rearrangements) are no longer damped to 0.25× | `geom_move_weight`=2.0 | ✅ verified |
| Huber δ 0.5→1.0 | Keeps large reactive-bond misses in the quadratic regime so they get real gradient instead of the flat/hedged linear region | `geom_huber_delta`=1.0 | ✅ verified |
| Uncertainty head + Kendall-Gal attenuation | Per-atom log-variance from EGNN features; `exp(−s)·huber + 0.5·s` lets the model express uncertainty instead of hedging every pair, and revives the dead UQ | `geom_uncertainty`=True | ✅ verified |
| Cosine LR schedule | Replace `PlateauWarmupScheduler` (40-epoch warmup, then sat at peak LR 100+ epochs before its first plateau decay) with a 5-epoch linear warmup + cosine anneal to `1e-6` over the epochs until `swa_start`. Continuous decay lets the model settle into sharper minima instead of overshooting. | `warmup_epochs`=5 | ✅ verified |
| Per-molecule unpadded differentiable MDS | Restore end-to-end gradient through the 3D reconstruction — see resolution below | `mds_differentiable`=True | ✅ verified |
| Loud non-finite loss/grad guard | Detect NaN/inf loss *or* gradients (a finite loss can still yield NaN grads from the differentiable-MDS `eigh` backward), skip the step, log it, and report per-epoch skip counts — instead of the silent `nan_to_num` forward mask (which never caught backward NaNs anyway). Removes the in-graph MDS `nan_to_num`. | (always on) | ✅ verified |
| **Ea head: geometry-trust inputs (mode B)** | Feed **detached** geometry-trust signals into the Ea head: EGNN refinement displacement \|x_ts − x_init\| (mean/max/RC-region — an asynchronicity proxy) + geometry log-variance (mean/RC) when `geom_uncertainty` is on. The Ea head reads detached TS features and was otherwise blind to whether that geometry is trustworthy; this gives it that signal without the Ea gradient reshaping geometry. **Ea stays on `SmoothL1` (no NLL switch)** — trust is an input feature, not a reported confidence. Fed through a **dedicated, per-channel running-normalized, zero-init encoder** added as a residual to the physics embedding (NOT concatenated into the shared `phys_enc`) — see note. | `ea_use_geom_trust`=True | ✅ verified |

**Resolved (formerly a "proven dead end"):** `mds_differentiable` — letting the geometry head learn
*through* the 3D reconstruction by backpropagating through the MDS `eigh`. The **original padded** path was a
**silent no-op**: the dummy-shift that separates padded atoms creates degenerate eigenvalues, `eigh`'s
backward goes non-finite, and the sanitizer zeroes it — so with any padded molecule (i.e. always, 9–30 atoms
in a 30-slot tensor) the geometry head received **zero** gradient. **Fix:** `torch_mds_coords(differentiable=True)`
now slices each molecule to its real `[n, n]` block and runs `eigh` **unpadded**, so there is no
padding-induced degeneracy and the backward is stable — gradient flows geom-loss → EGNN → MDS → GeometryHead.
Two correctness details were required to make it real rather than another silent no-op: (1) the seed passed to
MDS must **not** be detached when the flag is on (it now isn't); (2) `sqrt(λ)` on a clamped eigenvalue has an
**infinite backward gradient at 0**, and early in training the dim-th eigenvalue sits at/near zero, so a bare
`.sqrt()` re-emits `inf`/`NaN` grads that the forward `nan_to_num` cannot catch — a `+1e-8` floor before the
`sqrt` keeps the backward finite. **Now default on.**

**Caveats on the MDS fix.** (a) *Throughput* — the per-molecule loop replaces one batched `eigh` with a
Python loop of `.item()` syncs + tiny per-molecule `eigh` calls; benchmark before a long run. (b) *Residual
degeneracy* — **fixed at the source (2026-07-20).** The `+eps`/sqrt fix removed the `sqrt(0)` NaN, but
`eigh`'s *backward* still carried `1/(λ_i − λ_j)` eigenvector-coupling terms that blow up to inf/NaN when the
dim-th eigenvalue collides with the near-zero cluster (predicted distances not yet a valid 3-D Euclidean
matrix). The `+1e-6` diagonal jitter breaks the forward degeneracy but leaves the backward gap ~1e-6, so the
gradient reaches ~1e6–1e11 and overflows fp16 after GradScaler. This is now solved by `_StableEigh`, a custom
autograd `eigh` whose backward replaces `1/(λ_i − λ_j)` with the Lorentzian-broadened
`(λ_i − λ_j)/((λ_i − λ_j)² + ε)` (`ε = 1e-4`, `_MDS_EIGH_BACKWARD_EPS`) — finite everywhere, capped at
`1/(2√ε) ≈ 50`, and numerically identical to the true gradient away from degeneracy (verified against
autograd `eigh` to ~1e-9). On a forced tight tie at the dim boundary the plain-`eigh` backward gives a
gradient of ~1.7e11; `_StableEigh` gives ~3.8. The loud non-finite loss/grad guard remains as a backstop, but
the eigh backward should no longer be the source of skips. (c) *Redundancy* — the coordinate-native
rewrite in the Appendix deletes MDS entirely and makes this bridge obsolete; if that migration is committed,
this fix is temporary.

**Geometry-trust encoder note (regression fix).** The first 6-epoch run with a *naive* concat of the raw
trust features into the shared `phys_enc` **regressed Ea by ~1.5–2 kcal/mol vs the previous run at the same
epochs** (geometry was unaffected / slightly better). Cause: the raw displacement channel (0–several Å, and
largest/noisiest exactly when geometry is bad early) swamped the z-scored energy descriptors and the dominant
`de_rxn`/BEP signal. Fix (implemented): a **dedicated trust encoder** with (i) per-channel running
normalization (BatchNorm-style buffers, robust to a size-1 tail batch) so no channel dominates by scale, and
(ii) a **zero-init final layer** so the trust stream contributes exactly 0 at init — the Ea head starts
byte-identical to the no-trust baseline (unit-tested) and self-gates the feature in only as gradients confirm
it helps. This should remove the early regression while preserving any late-training benefit.

**Success metric after retrain:** `pred_dev / true_dev` slope should move from **0.81 → ~1.0**; unimolecular
(1-fragment) median `dist_MAE` should drop from 0.117 toward the multi-fragment 0.05–0.08 range. For the Ea
geometry-trust change (mode B): the Ea fat tail should shrink (val RMSE 7.87 toward MAE 4.67) **concentrated
on high-displacement / high-uncertainty reactions** — if the tail shrinks uniformly instead, the trust
features aren't the mechanism.

---

## Drawbacks & pitfalls of the uncertainty / trust approach

The uncertainty head (`geom_uncertainty`) and the Ea geometry-trust inputs (`ea_use_geom_trust`) are genuine
improvements, but they carry well-known failure modes. Recording them here so they are not mistaken for a
cure.

### P1 — Neural networks are overconfident; a "trust" score can be a lie
Uncalibrated networks routinely assign high confidence to wrong outputs — the model can emit an invalid
geometry and still report a low variance for it. **Scope for us:** this bites hardest only if `geom_logvar`
is ever *reported* as a user-facing confidence (the report notes we revived a previously dead UQ estimate).
In the current design the log-variance and displacement enter the Ea head as **input features**, not as a
published confidence, so even a miscalibrated variance can still be a *useful predictor* the head learns to
read — #4 is deliberately robust to this. The moment the estimate is surfaced to a user or used to gate
decisions, it must be calibrated: check **coverage** (do the ±σ bars actually contain the true distances at
the stated rate?), report **ECE**, and apply temperature scaling. Current mitigations are weak — zero-init
log-var head (starts neutral, variance 1) + clamp `[-7, 7]` (prevents runaway) — and are *not* calibration.

### P2 — NLL/attenuation training instability ("cheating" the loss)
Switching a regression from MSE/Huber to a heteroscedastic NLL lets the model reduce the loss by predicting
**large uncertainty everywhere** instead of learning the target — often destabilizing the first ~50 epochs.
**Scope for us:** the Ea head was **deliberately kept on `SmoothL1`** — #4 adds trust as inputs and does *not*
switch Ea to NLL, so it introduces no NLL instability. The risk lives entirely in the **geometry**
`geom_uncertainty` head, whose Kendall-Gal attenuation `exp(−s)·huber + 0.5·s` is currently active **from
epoch 0**. Worse than generic instability: if the geometry head inflates `s` to dodge the `0.5·s` penalty, it
"gives up" on hard pairs by declaring them uncertain — **re-introducing the exact under-shoot the other four
fixes exist to remove.** Recommended mitigations (not yet implemented): (a) **NLL warmup** — pure Huber for
the first K epochs, then ramp in the attenuation; (b) **β-NLL** (Seitzer et al. 2022) — scale the NLL by a
*detached* `var^β` (β≈0.5) so variance cannot suppress the fitting gradient; (c) **monitor** mean predicted
variance per epoch — a monotone climb while train error stalls is the cheating signature.

### P3 — A trust score does not fix bad chemistry
Uncertainty is **diagnostic, not curative**. If the model fundamentally cannot represent a phenomenon, a
confidence estimate does not repair it — it only labels it. Two concrete cases in this pipeline:
**chirality** (distance matrices are chirality-blind by construction — see the Appendix) and residual **MDS
degeneracy**. Both are *architectural* and are addressed by architectural fixes (unpadded MDS for degeneracy;
the coordinate-native rewrite for chirality and the interpolation prior) — never by a trust score. One nuance
in #4's favour: the **displacement** feature measures *asynchronicity*, a real physical property and a proxy
for reaction class, so it is somewhat more chemistry-aware than a pure variance — but it still makes the model
*self-aware*, not *more accurate*. The trust/uncertainty work is **complementary** to the structural fixes
(useful for triage, active-learning sample selection, and letting the Ea head route around untrustworthy
geometry), and must not be treated as a substitute for them.

---

## Appendix — Scope: coordinate-native geometry (the real MDS-bottleneck kill)

The loss/uncertainty fixes attack the *symptoms* of the under-shoot. The **root** is the geometry
*algorithm*: `distance interpolation → detached MDS (eigh) → incremental EGNN`. Each stage is a
deterministic, midpoint-anchored, lossy transform, and the MDS detachment cannot be removed in the
padded-batch design (proven above). The SOTA line (React-OT 0.053 Å, OA-ReactDiff, LEFTNet on Transition1x)
skips all of it: an **equivariant network predicts TS coordinates directly from R+P coordinates**,
end-to-end, no eigh. Scoping that migration here.

### Goal
Replace the distance→MDS→EGNN chain with a coordinate-native, end-to-end-differentiable geometry head that
predicts TS coordinates from reactant/product coordinates, and supervise it directly against the DFT TS
geometry `c_TS`.

### What changes

**1. Data layer (`ReactionDataset`, `move_batch_to_device`).** Currently the model sees only distance
matrices (`D_R/D_I/D_P`). Add the raw coordinates `c_R`, `c_P` (already in every `sample` dict, and already
used by the coord-noise branch and by `xtb_qm_validation`) to the batch. Define a **common reference frame**:
Kabsch-align the product onto the reactant, giving `c_P_aligned`; the physical midpoint
`c_I = (c_R + c_P_aligned)/2` is the same construction inference already uses as the MDS reference — reuse it
as the coordinate seed. Handle multi-fragment cases with the existing `kabsch_align_reactant_fragments`.

**2. Geometry head.** Drop `GeometryHead` (distance interpolation) and `torch_mds_coords` (eigh MDS). Seed
the existing E(n)-equivariant `EGNN` with the real `c_I` midpoint coordinates (not the MDS of a predicted
distance matrix) and let it predict the displacement to the TS. Node features still come from `PSICore`.
This removes both the interpolation prior *and* the eigh bottleneck; gradient flows end-to-end through
coordinates. (The EGNN is already E(n)-equivariant, so no new equivariant machinery is needed — it just
operates on a real seed instead of an eigen-embedded one.)

**3. Equivariance / frame.** The distance-matrix design bought SE(3)-invariance for free; coordinate-native
must be explicit. Two safe options: (a) predict in the reactant frame and train with a **Kabsch-aligned**
coordinate loss to `c_TS` (frame-invariant target); or (b) keep the equivariant EGNN and let equivariance
handle it, with the alignment only defining the seed. Distance-matrix auxiliaries (triangle, steric) still
apply — compute them from the predicted coordinates. **Bonus:** coordinates recover chirality/handedness,
which distance matrices throw away.

**4. Loss.** Primary: Kabsch-RMSD (or aligned-coordinate Huber) of predicted coords vs `c_TS`. Keep the
movement-aware weighting, triangle, steric, and spectator terms (recomputed from coords). The uncertainty
head carries over unchanged (per-atom coordinate variance).

**5. Ea head: unchanged.** It reads the EGNN's node features `h_ts` (detached) — that interface is preserved,
so the energy path and the two-stage training are untouched.

### Risks & mitigations
- **Reference-frame / alignment stability** — Kabsch is standard and stable; align once per sample, cache.
- **Equivariance bugs** — unit-test that a random rotation of R+P rotates the predicted TS identically.
- **Training instability from a bigger change** — gate behind `geom_mode: "coord" | "distance"` so the old
  path stays runnable; retrain from scratch (weights are architecture-specific).
- **Multi-fragment framing** — reuse `kabsch_align_reactant_fragments`; the fragment logic already exists.

### Effort & payoff
Medium–high: touches the data layer, the geometry head, and the geometry loss, but **keeps the PSICore
backbone, the EGNN module, and the entire Ea head.** Expected payoff: removes the under-shoot at its root
(no midpoint interpolation prior, no lossy MDS), recovers chirality, and adopts the exact inductive bias of
the current SOTA geometry models — the single change most likely to close the gap to React-OT-class
`dist_MAE`.

---

## Caveats

- `D_midpoint` uses `D_I = (D_R_aligned + D_P)/2`; the interpolation head mixes `D_R`/`D_P` with a learned
  `α`, so `maxdev` is a close but not exact proxy for the head's reachable set. The correlation is strong
  enough that the conclusion holds regardless.
- The xTB cross-reference covers 1,069 of the 6,000 val reactions; the "always-fail" set will grow as the
  remaining ~4,900 are validated.
- Overfitting statistics are from a single archived run (2026-07-19); the structural conclusions (mode A)
  are dataset properties and should be run-independent.
