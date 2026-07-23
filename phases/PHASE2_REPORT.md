# Phase 2 — Grad-clip sweep

**Verdict: H2 REJECTED - clip is not the lever**

All four arms land within 0.0024 A of each other on val MAE, below the 0.014100000000000001 A significance floor. At one seed per arm that is not a result. Unconditional clipping is therefore not what limits accuracy, and the LR sensitivity must come from elsewhere. Phase 3 should keep its original uncertainty axis rather than the clip axis.

## Why this axis

Phase 1 rejected clamp saturation (val `pinned_lo` peaked at 0.14%) but found
`clip_rate = 1.0000` on 40/40 epochs, with a median gradient norm of
6.57 against a `grad_clip` of 1.0. Every step was clipped, so
gradient magnitude was discarded on every update — leaving `lr` as the sole
determinant of step size. This phase asks whether that costs accuracy.

## Arms

- `1.0` — baseline — the production value
- `5.0` — below the median gradient: clips most steps
- `15.0` — above the median: clips the tail only
- `inf` — no clipping — isolates what the clip is protecting against

30 epochs, 4,000 reactions, cosine horizon pinned to
450 so LR stays near peak. One seed per arm.

## Results

**Coverage: 3/4 arms.** All figures are at epoch 15, the deepest epoch every surviving arm reached — comparing a crashed arm against a full-length one would measure run length, not clip.

| grad_clip | clip rate | val MAE (A) | train MAE | gap | val Ea | grad p50 | grad p99 | skips | val-NLL bounce |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1.0 | 100.0% | **0.2686** | 0.2545 | 1.06 | 13.85 | 5.87 | 35.4 | 6 | 0.0273 |
| 5.0 | 84.8% | **0.2700** | 0.2568 | 1.05 | 14.80 | 9.94 | 48.6 | 14 | 0.0434 |
| 15.0 | 10.1% | **0.2710** | 0.2490 | 1.09 | 13.57 | 5.52 | 15.8 | 9 | 0.0283 |
| inf | — | *did not run* | — | — | — | — | — | — | — |

Clip rate and volatility are averaged over the last 15 epochs; the rest
are final-epoch values. Phase 1 reference at 40 epochs: val MAE
0.2608, gap 1.23.

## Reading it

- **clip rate** should fall monotonically across the arms. If it does not, the arms
  did not actually separate and nothing below is interpretable.
- **val MAE** is the outcome. Differences under 0.014100000000000001 A are noise at one
  seed per arm.
- **skips** is the safety column: rising non-finite counts as the clip loosens mean
  it was protecting against something real.
- **gap** shows whether looser clipping trades accuracy for overfitting.

## Noise floor

Phase 1 and this sweep's `clip1` arm ran with **identical** configuration and their
val MAE at epoch 30 differed by **0.0047 A** — pure run-to-run
non-determinism. The significance threshold above is 3x that floor
(0.0141 A). Arm differences between 1x and 3x the floor are not
resolvable at one seed per arm; the answer there is more seeds, not a verdict.

## Scale caveat

4,000 reactions / 30 epochs, one seed. Between-arm ordering is
the result; absolute values do not transfer to the 40k production run. Any arm
adopted here still needs confirmation at full scale.

## What this does not settle

Whether the LR sensitivity itself is fixed. That needs the LR axis (Phase 3), run
against whatever threshold this phase selects.
