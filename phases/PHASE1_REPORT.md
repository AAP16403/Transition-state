# Phase 1 — Log-variance observation run

**Verdict: H1 REJECTED**

Only 0.1% of val atoms are at the clamp, below the 5% rejection threshold. The head is NOT saturating, so the exp(-7)=1097x amplification story cannot explain the volatility. Do not build Phase 2 -- re-diagnose first. The val log-variance distribution (p1/p50/p99 below) is the place to start.

## What was run

A single 40-epoch run on 4,000 reactions, cosine horizon pinned to
450 epochs so the LR stays near peak throughout (an uncompressed
schedule is required to observe high-LR behaviour in a short run). Early stopping
disabled. Completed 40 epochs.

## Scale caveat

Absolute values at 4,000 reactions do not transfer to the 40k production
run — the train/val gap opens faster and wider with less data. The full-scale column
below is for orientation only; the verdict rests on the *pattern* (saturation present
or absent, volatility ratio large or small), not on matching numbers.

## Log-variance distribution — the measurement

| quantity | train | val |
|---|---:|---:|
| fraction pinned at -7 clamp | 0.1% | **0.1%** |
| fraction pinned at +7 clamp | — | 0.0% |
| logvar p1 | — | -6.25 |
| logvar p50 | -2.85 | -2.75 |
| logvar p99 | — | 0.75 |

val pinned_lo moved 0.0% → 0.1%
across the run (rising).

A pinned atom contributes `exp(7) ≈ 1097x` amplification to its pairs' gradient and
loss. The pinned fraction is therefore the single number that determines whether the
NLL is a well-behaved objective or a lever arm on a few outliers.

## Volatility — did the signature reproduce?

| metric | this run | full-scale ref (ep 100-147) |
|---|---:|---:|
| BOUNCE — mean \|Δ val NLL\|/epoch | 0.1049 | 0.1860 |
| SMOOTH — mean \|Δ val MAE\|/epoch | 0.00216 | 0.00082 |
| **RATIO — BOUNCE / SMOOTH** | **49x** | 227x |
| mean \|Δ train NLL\|/epoch | 0.0155 | 0.0048 |
| GAP — val MAE / train MAE | 1.23 | 1.66 |

Both volatility figures are averaged over the last 15 epochs. The train-NLL
row is the control: if train is smooth while val bounces, the instability is
generalization, not optimization.

## Gradient scale

| quantity | value |
|---|---:|
| pre-clip grad norm p50 | 6.631 |
| pre-clip grad norm p99 | 17.254 |
| **tail ratio p99/p50** | **2.6x** |
| steps where the clip bit | 100.0% |
| non-finite grad skips (all epochs) | 8 |

Recorded on steps that actually took an update; skipped steps contribute no norm.

**CLIP SATURATION — 100.0% of steps were clipped.** The median gradient norm is 6.63 against a `grad_clip` of 1.0, so the threshold sits 6.6x below the *median* rather than catching outliers. Every update is therefore a fixed-norm direction with its magnitude discarded, removing the "gradients are shrinking, we are near a minimum" signal and leaving `lr` as the sole determinant of step size — a coherent mechanism for larger LR converging worse. Genuine outliers do exist (p99 17, 8 non-finite skips), so some clipping is load-bearing. Phase 2 sweeps the threshold to separate the two.

## Trend

| epoch | val NLL | val MAE (A) | train pin_lo | val pin_lo | val logvar p50 | grad p99/p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | -0.3861 | 0.3076 | 0.1% | 0.0% | -1.75 | 44.5 |
| 5 | -0.5828 | 0.2816 | 0.0% | 0.0% | -1.75 | 2.3 |
| 9 | -0.6218 | 0.2734 | 0.0% | 0.0% | -1.75 | 3.5 |
| 13 | -0.6367 | 0.2706 | 0.0% | 0.0% | -1.95 | 3.1 |
| 17 | -0.6193 | 0.2714 | 0.0% | 0.0% | -1.95 | 4.3 |
| 21 | -0.4599 | 0.2691 | 0.0% | 0.0% | -2.55 | 2.8 |
| 25 | -0.6200 | 0.2670 | 0.0% | 0.0% | -2.15 | 2.1 |
| 29 | -0.4872 | 0.2674 | 0.0% | 0.0% | -2.45 | 2.3 |
| 33 | -0.3485 | 0.2617 | 0.0% | 0.0% | -2.85 | 2.1 |
| 37 | -0.4868 | 0.2655 | 0.0% | 0.0% | -2.55 | 2.7 |
| 40 | -0.3044 | 0.2608 | 0.1% | 0.1% | -2.75 | 2.6 |

## What this does and does not settle

Settles: whether the log-variance head saturates at the clamp, and whether the
val-NLL volatility signature reproduces at reduced scale.

Does not settle: whether saturation *causes* the volatility (Phase 2, clamp sweep)
or whether it drives the LR sensitivity (Phase 3, LR × uncertainty factorial).
Those need ablations; this run is observation only.
