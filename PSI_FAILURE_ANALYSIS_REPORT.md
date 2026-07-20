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

## Caveats

- `D_midpoint` uses `D_I = (D_R_aligned + D_P)/2`; the interpolation head mixes `D_R`/`D_P` with a learned
  `α`, so `maxdev` is a close but not exact proxy for the head's reachable set. The correlation is strong
  enough that the conclusion holds regardless.
- The xTB cross-reference covers 1,069 of the 6,000 val reactions; the "always-fail" set will grow as the
  remaining ~4,900 are validated.
- Overfitting statistics are from a single archived run (2026-07-19); the structural conclusions (mode A)
  are dataset properties and should be run-independent.
