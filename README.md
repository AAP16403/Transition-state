# PSI — Transition State Prediction

Predicts the 3D geometry and activation energy (Ea) of a chemical reaction's
transition state from the reactant and product geometries alone. Trained on RGD1
(40,000 CHNO reactions).

---

## The pipeline

**`psi_cloud_pipeline.py` is the pipeline.** One standalone file (~1,580 lines),
one code path, targeting Kaggle T4 x2.

It replaces three divergent copies. The previous main file,
`psi_full_pipeline.py`, had grown to ~5,400 lines carrying every mode it had ever
tried — distance *and* coordinate geometry, distance *and* xTB reaction centres,
MDS, geom-only/ea-only staging, triangle losses — all live behind config flags.
Every fix then had to be applied in several places and re-verified per mode. A
single audit pass found **five wiring gaps that existed only because a superseded
branch was still switched on**, including one that would have crashed a 47-hour
run four minutes in.

The rewrite's rule: **delete superseded paths, do not gate them.** A knob nobody
turns is a branch nobody tests.

| | |
|---|---|
| No config dict, no argparse hyperparameters | Hyperparameters are module constants |
| No `try`/`except`, no fallback branches | Anything that cannot proceed correctly raises |
| Deleted, not disabled | GeometryHead interpolation prior, MDS/eigh, coarse-distance aux loss, both triangle losses, delta-clamp, coord-noise, geom-only/ea-only staging, distance-rule spectator mask |

### Status of the other files

| File | Status |
|---|---|
| `psi_cloud_pipeline.py` | **Active.** All new work goes here. |
| `psi_full_pipeline.py` | Legacy. Frozen except for correctness fixes. Currently holds the only trained checkpoints. Retire once the cloud pipeline reproduces its numbers. |
| `psi_swarm_pipeline.py` | Legacy, unmaintained (Mixture-of-Experts experiment). |
| `geom_diagnostics.py`, `resume_evaluation.py`, `build_bond_orders.py` | Support tools for the legacy pipeline. Their function is folded into the cloud pipeline. |

---

## Running it on Kaggle

Works on **GPU P100** (single, in-process) or **GPU T4 x2** (DDP). Turn
**Internet ON** — the first run streams RGD1 (~1.34 GB) from Figshare into
`/kaggle/working/RGD1_Dataset` and parses it into `samples_cache.pkl`. Save the
output as a Dataset and attach it thereafter so that happens exactly once.

Use `!python psi_cloud_pipeline.py <stage>` from a cell. `%run` also works:
IPython injects its own `sys.argv` (`-f /root/.../kernel-abc.json`), which the
launcher ignores rather than mistaking for a stage name.

Three stages. The first is CPU-only and runs once per dataset.

**Stage 1 — bond orders** (Internet ON, accelerator OFF, ~1.5–2 h on 4 cores):

```bash
!pip install tblite
!python psi_cloud_pipeline.py bond-orders
```

Writes `bond_orders_cache.pkl`. Save the notebook output as a Kaggle Dataset and
attach it to every training session — recomputing it would burn most of a GPU
quota on CPU work.

**Stage 2 — training** (accelerator: GPU T4 x2):

```bash
!python psi_cloud_pipeline.py train
```

Kaggle sessions cap at ~9–12 h, far short of a full run, so **every epoch
checkpoints and training resumes automatically** from `psi_latest.pt`. Save the
output as a Dataset, attach it to the next session, re-run the same command.

**Stage 3 — evaluation** (GPU, minutes):

```bash
!python psi_cloud_pipeline.py eval
```

Scores `psi_best.pt` per reaction into `results.json`: distance MAE and
coordinate RMSD as **mean and median**, Ea error, and the chirality-flip rate.
Separate from training on purpose — a session is cut long before the epoch budget
runs out, so an evaluation that only fired after the last epoch would never run.

### Per-epoch telemetry

The epoch line reports the two failure modes the legacy runs actually hit, not
just the headline metrics:

```
  142 | trGeom 0.0969 | vaGeom 0.1565 | gap 0.0596 | vaRMSD 0.412A | flip  0.3%
      | trEa  3.968 | vaEa  5.226 | lvFloor 18.1% | clip 100% | lr 1.24e-04 | 231.4s *
```

`gap` is the overfitting margin (flaw 1), `lvFloor` the fraction of atoms pinned
at the log-variance floor (flaw 2), `flip` the chirality-flip rate, `clip` the
fraction of steps hitting the gradient clip (flaw 6).

RGD1 (`RGD1_CHNO.h5`, `DFT_reaction_info.csv`) is pulled from Figshare on first
use, or picked up from any attached Kaggle dataset.

### What it uses the GPU for

DDP across both T4s via `mp.spawn`, or in-process on a single GPU · fp16 AMP +
GradScaler · `torch.compile` (inductor) · fused AdamW · TF32 + cuDNN autotune ·
pinned memory with non-blocking transfers · persistent dataloader workers with
prefetch. Validation is sharded by slicing rather than `DistributedSampler`,
which pads the last shard with duplicates and would double-count reactions in the
selection metric.

Two hardware facts drive this, both checked at startup rather than assumed:
**neither P100 (Pascal) nor T4 (Turing) supports bf16**, so fp16 is the only
option; and **Triton requires sm_70+, so `torch.compile` cannot run on P100**
(sm_60) — it is enabled on T4 and skipped on P100. The banner prints what is
actually active.

---

## Architecture

```
(D_R, D_I, D_P) + atom identity
        │
        ▼
   PSICore  ── gaussian RBF → per-atom GRU over (R, midpoint, P) → transformer
        │
        ▼  node features
   EGNN  ◄── seeded with the real Kabsch-aligned R/P midpoint  (NOT an MDS embedding)
        │
        ├──► TS coordinates ──► distances
        ├──► per-atom log-variance
        └──► EaHead (detached features + geometry-trust)
```

**Coordinate-native geometry.** The EGNN starts from the true R/P midpoint and
predicts TS *coordinates*, supervised against the DFT structure through a
Kabsch-aligned Huber loss. The old path predicted a distance matrix as
`α·D_R + (1−α)·D_P + δ` and embedded it via MDS. The geometry failure atlas
attributed **45.9 % of failing validation reactions** to that interpolation prior
(`geom_head_interpolation_bound`) and only 0.2 % to MDS loss — while MDS cost
**58 % of the forward pass**. Coordinate-native removes an accuracy ceiling and a
throughput cost at the same time, and it is the only formulation that can
represent chirality at all: a distance matrix cannot distinguish an enantiomer.

The seed uses **one global Kabsch transform**, never per-fragment alignment.
Per-fragment alignment lands each reactant fragment on its *product* fragment's
centroid, overwriting the reactant's inter-fragment arrangement — and
`cross_fragment_orientation` is 31.9 % of failing reactions.

**xTB reaction centres.** Reactive atoms come from cached GFN2-xTB Wiberg bond
orders, not a covalent-radius cutoff. The cutoff is blind to bond *order* — a C–C
single and a C=C double bond both sit far inside the threshold — and misses 8,754
reactive atoms, giving a different reactive-atom set in 22.0 % of reactions. The
xTB spectator mask also fixes the physics-informed midpoint target: measured
|D_TS − D_I| is **0.0143 Å** on the pairs it selects against **0.1218 Å** for the
distance rule.

**Heteroscedastic uncertainty.** A per-atom log-variance head with Kendall-Gal
attenuation lets the model declare a pair hard instead of hedging every
prediction toward the midpoint. Its output also feeds the Ea head as *detached*
geometry-trust features (refinement displacement, log-variance), so the Ea head
knows how much to trust the geometry it is reading without ever reshaping it.

**Loss.** Kendall-Gal-attenuated Huber on distances with reaction-role and
movement-aware pair weighting, plus the Kabsch coordinate loss, plus spectator
midpoint and steric-floor physics terms, plus risk-weighted geometry and Ea
terms. The Ea head always reads `h_ts.detach()`, so its gradient never reaches
geometry.

---

## Current results

From the still-running legacy job (`psi_full_pipeline.py`, distance mode), epoch
143 of 800:

| | train | val |
|---|---:|---:|
| Distance MAE | 0.0969 Å | **0.1565 Å** |
| Ea MAE | 3.97 kcal/mol | **5.23 kcal/mol** |

The last **converged** run (2026-07-19), measured from the per-reaction records
in `psi_results_dashboard.html`:

| | train | val |
|---|---:|---:|
| Distance MAE (mean) | 0.0541 Å | **0.1241 Å** |
| Distance MAE (median) | 0.046 Å | **0.0983 Å** |

For reference, literature median D-MAE: Choi ≈ 0.095 Å, TSDiff (single sample)
≈ 0.137 Å. The median is competitive; **the mean is not**, and the gap between
them is the story — a minority of structurally hard reactions carries the error.

> The cloud pipeline has **not** been trained yet. It is verified structurally
> and numerically on synthetic data, not convergence-tested. Every number above
> comes from the legacy pipeline.

---

## Known flaws

Ordered by how much they matter.

**1. Overfitting is the dominant error source and nothing currently addresses it.**
The train/val gap widens monotonically with no inflection: 0.006 Å at epoch 5 →
0.029 at epoch 60 → 0.047 at epoch 100 → **0.060 at epoch 143**. Train MAE falls
roughly 2.3× faster than validation. Dropout (0.1), weight decay (1e-3) and
coord-noise (0.0) are unchanged from the run that converged to a 2.3× gap. Raising
`coord_noise` is **not** the fix — it invalidates the sample cache; a decoupled
seed-noise or dropout schedule is the right lever.

**2. The uncertainty head is drifting toward its floor.**
`pinned_lo` — the fraction of atoms clamped at the minimum log-variance — climbs
linearly: 0 % → **18.1 % by epoch 143**, roughly 1 % per 10 epochs, with median
log-variance marching −1.55 → −5.35 and no plateau. Extrapolates to ~48 % by the
SWA handover. The NLL is being reduced partly by shrinking σ rather than by
improving accuracy. An earlier phase check rejected clamp saturation at 0.14 %,
but that ran 4,000 reactions for 40 epochs and does not cover this regime.

**3. Returns are nearly exhausted well before the epoch budget.**
Validation improvement per 20-epoch window decays geometrically (ratio ≈ 0.65):
−0.0119, −0.0076, −0.0047, −0.0032 Å. Naive asymptote ≈ 0.153 Å. That fit cannot
see the cosine anneal or the SWA phase, both of which usually deliver a step
change — but 800 epochs is not obviously worth its cost.

**4. Chirality has never been measured — now instrumented, still unknown.**
The legacy geometry atlas reports it as `NOT MEASURED`. Distance MAE is
chirality-blind, so a model that learned every enantiomer would score a perfect
distance MAE. The cloud pipeline now measures the flip rate every epoch by
comparing a proper-rotation superposition against a reflection-allowed one — both
from the same SVD, so it costs nothing — and reports it per reaction at eval. The
number itself is still unknown until the first real run.

**5. `--rc-source xtb` re-deals the train/val split.**
The stratified split bins by list *index*, so dropping the 78 reactions xTB
cannot describe reshuffles every assignment — not just those 78. An xTB run
therefore does not share a validation set with a distance run at the same seed.
The dropped ids are recorded in `split_diagnostics.json` so runs can be compared
on the intersection.

**6. Every gradient step is clipped.** `clip_rate = 1.00` on all 143 epochs,
median gradient norm ≈ 7.8 against `grad_clip = 1.0`. Investigated and dismissed:
a clip sweep (1.0 / 5.0 / 15.0) moved validation MAE by 0.0024 Å, below the
0.0141 Å significance floor. Adam is approximately scale-invariant to a uniform
rescale. Not a lever.

**7. Occasional non-finite gradients.** One batch out of ~708 in ~25 % of epochs,
flat since epoch 21 — not escalating. Likely the degenerate-SVD edge case in the
displacement normalisation. The cloud pipeline puts the epsilon *inside* the
sqrt, which should remove it.

---

## Fixes applied 2026-07-22

**Legacy pipeline** (`psi_full_pipeline.py`) — five wiring gaps, all from
superseded branches still being live:

1. `--rc-source xtb` was only half-applied. The Ea head and geometry-trust
   features still used the covalent-radius rule while the loss used xTB — two
   reaction-centre definitions in one run, which the code elsewhere explicitly
   raises to prevent.
2. **Coordinate mode silently discarded 33 % of the model.** `PSICore`
   (2,300,096 params) had exactly one consumer, the geometry head, which coords
   mode does not build. It ran every forward, received zero gradient and was
   thrown away — and the EGNN saw no reaction context at all. Now the encoder
   feeds the EGNN node features. Measured `core` gradient: 8e-6 → 0.061.
3. `geom_diagnostics.py` crashed in coords mode (missing `c_seed`, and
   `D_coarse` is `None`), so a coords run would finish with no diagnostics.
4. `predict` built its seed with per-fragment alignment while training used
   global — a train/inference frame mismatch.
5. **76 reactions have an empty xTB reaction centre** and the Ea head hard-raises
   on one. Fixing (1) would have turned that latent data issue into a crash
   inside epoch 1. Both failure classes are now dropped together, loudly.

**Cloud pipeline** — three bugs caught by testing the new code:

- `dist2.sqrt()` in the EGNN coordinate update produced NaN gradients into the
  encoder. `d/dr √r` is infinite at zero, and `dist2` is exactly zero on every
  self-pair, every padded pair, and every edge at initialisation. Epsilon now
  lives *inside* the sqrt.
- SWA read `swa_model.module` before `update_parameters`, so the first SWA epoch
  would have scored the untrained initialisation.
- `mp.spawn` would have pickled all 40k samples to every rank (>1 GB, paid
  twice). Workers now load from the on-disk cache.

A follow-up audit against the flaw list above then found the pipeline could not
*observe* several of the problems it was built to fix:

- **The log-variance histogram spanned ±8 while the clamp is ±7**, so bin 0 was
  permanently empty and `pinned_lo` would have read 0 % however hard the head was
  pressed against its floor — silently hiding flaw 2. Range now equals the clamp.
- No log-variance, gradient-norm or clip-rate telemetry existed at all (flaws 2
  and 6 were unobservable). Now logged per epoch, DDP-reduced.
- No chirality measurement (flaw 4), despite handedness being much of the reason
  to predict coordinates.
- `RESULTS_PATH` was declared and never written: there was **no per-reaction
  output**, so the mean-vs-median split that is the whole story of the current
  results could not be computed. Added as the `eval` stage.
- The dropped-reaction ids were printed but not persisted, making an
  intersection comparison against a distance-mode run impossible after the fact.
- `HARTREE_TO_KCAL` was dead. A test now asserts no module constant is declared
  and never read.

A third pass over correctness (rather than features) found four more:

- **The tblite call dropped the spin-summation.** tblite can return
  `[n, n, nspin]`; the proven implementation sums over spin. Without it every
  downstream `[n, n]` index is wrong — discovered at the *end* of a ~2 h
  precompute.
- **One SCF failure would have destroyed the whole precompute.** GFN2-xTB fails
  to converge on ~6 of 40,000 geometries and cannot be asked in advance. This is
  the file's single deliberate `try`/`except`, narrowly scoped: the failure is
  recorded and the reaction *excluded*, never patched with a substitute (a zero
  bond-order matrix would read as "every bond broke").
- **The Ea head had no empty-reaction-centre guard.** An all-false mask makes
  every softmax logit equal, so the "reaction-centre pool" silently degrades into
  a whole-molecule mean. That guard is exactly how the 76 empty-xTB-centre
  reactions were found in the first place; without it they would train quietly on
  the wrong feature.
- **Resume dropped the SWA scheduler state**, so continuing inside the SWA phase
  restarted SWALR's anneal and re-raised the learning rate that weight averaging
  is supposed to hold steady.

---

## Roadmap

**Now — validate the cloud pipeline.** Run stage 1, then a short training session
on Kaggle. Confirm it reaches the legacy pipeline's trajectory. It has never been
trained; everything so far is static verification.

**Next — retire the legacy files.** Once the cloud pipeline reproduces
val D-MAE ≈ 0.124 Å, delete `psi_full_pipeline.py` and `psi_swarm_pipeline.py`
rather than leaving them to drift.

**Then — attack the gap, which is where the remaining accuracy is.** In order:
a decoupled seed-noise / dropout schedule for flaw 1; a log-variance floor
penalty or a wider clamp for flaw 2; and a decision on the epoch budget once the
anneal + SWA contribution is actually measured rather than extrapolated.

**Then — read the chirality number.** The flip rate is now instrumented but has
never been observed on real data. If it is non-trivial, it is a failure mode no
previous run could even see.

**Not carried over — single-molecule prediction.** The legacy pipeline had a
`predict` path that read Gaussian `.log` files. It is deliberately absent: it
needs the log-file parser, and Kaggle is not where one-off predictions get run.
Add it back only when there is a reason to.

**Open question — the mean/median split.** Median D-MAE is competitive with
published work and the mean is not, so a structurally hard minority dominates.
The failure atlas points at cross-fragment orientation (31.9 %) and unimolecular
rearrangements. Coordinate-native geometry is the bet that this is fixable; that
bet is untested.

---

## Repository layout

```
psi_cloud_pipeline.py       THE pipeline — Kaggle T4 x2, two stages
psi_full_pipeline.py        legacy, frozen; holds current checkpoints
psi_swarm_pipeline.py       legacy, unmaintained
geom_diagnostics.py         per-sector geometry failure atlas (legacy)
resume_evaluation.py        rebuild outputs from psi_best.pt without retraining
build_bond_orders.py        standalone xTB bond-order precompute (legacy)
phases/                     controlled hypothesis tests (clip sweep, LR sweep)
PSI_FAILURE_ANALYSIS_REPORT.md   failure-mode analysis behind the design
PSI_PIPELINE_FINAL_REPORT.md     detailed pipeline report
RGD1_Dataset/               RGD1_CHNO.h5 + DFT_reaction_info.csv
```
