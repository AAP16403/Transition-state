# Phase 3 — Learning-rate sweep

**Verdict: H3 INCONCLUSIVE - budget-limited**

1 of 3 arm(s) were still improving at epoch 30 (lr=4.5e-4), so this sweep compared how far each learning rate got in a fixed epoch budget, not where each one converges. A faster LR trivially gets further under that constraint. The claim under test concerns converged runs (the originals ran 147 and 323 epochs), so this cannot settle it either way.

For the record the ordering is the OPPOSITE of the original impression: best val MAE improves with LR, spanning 0.0205 A (best 0.2521 at lr=4.5e-4), above the 0.0141 A floor. But the overfitting gap also rises with LR (1.10 -> 1.23), which is precisely how a faster LR could still finish worse once every arm has converged. Re-run to convergence, or match arms on training progress rather than epoch count.

## Why this is the direct test

The original "bigger LR is worse" impression came from two production runs differing
in ~8 dimensions at once, so LR was never isolated. Two candidate mediators have been
measured and killed:

| phase | hypothesis | result |
|---|---|---|
| 1 | log-variance clamp saturation | REJECTED — val `pinned_lo` peaked at 0.14% |
| 2 | grad-clip saturation | REJECTED — clip rate 100%→10% moved val MAE 0.0024 A |

Phase 2's negative is expected in hindsight: Adam is approximately scale-invariant,
so the uniform rescale that global-norm clipping applies is largely absorbed by its
`m/sqrt(v)` normalisation. This phase therefore tests the claim itself rather than
hunting further mediators.

## Arms

- `5e-5` — one third of production
- `1.5e-4` — baseline — the production value
- `4.5e-4` — 3x production

30 epochs, 4,000 reactions, cosine horizon pinned to
450 so the schedule is NOT compressed by the short run — every arm
holds near its peak LR throughout, which is the regime under test. One seed per arm.

## Results

| lr | val MAE (A) | best val MAE | train MAE | gap | val Ea | grad p50 | clip rate | skips |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 5e-5 | **0.2766** | 0.2727 | 0.2523 | 1.10 | 13.27 | 6.57 | 100.0% | 12 |
| 1.5e-4 | **0.2647** | 0.2647 | 0.2268 | 1.17 | 13.34 | 7.04 | 100.0% | 8 |
| 4.5e-4 | **0.2639** | 0.2521 | 0.2144 | 1.23 | 12.67 | 5.29 | 100.0% | 7 |

`best val MAE` is the minimum over all epochs, not just the final one: a high-LR arm
can oscillate around a good minimum without landing on it at epoch 30, and
reading the final epoch alone would misreport that as worse.

## Noise floor

Two identically-configured runs (Phase 1 and Phase 2 `clip1`) differed by
**0.0047 A** in val MAE at epoch 30 — pure cuDNN/AMP non-determinism. The
significance threshold is 3x that (0.0141 A). Differences between 1x
and 3x the floor need more seeds, not a verdict.

## Scale caveat

4,000 reactions / 30 epochs, one seed per arm. Between-arm
ordering is the result; absolute values do not transfer to the 40k production run.
Any LR adopted here needs confirmation at full scale.
