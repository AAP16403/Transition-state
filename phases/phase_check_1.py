#!/usr/bin/env python3
"""Phase 1 — observation run: is the geometry log-variance head saturating?

Background
----------
The full run (epochs 100-147) showed `val_geom` moving 0.186 nats per epoch while
`val_geom_mae_A` moved 0.00082 A -- the NLL is ~225x more volatile than the MAE it
is built from, and ~39x more volatile than the *train* NLL of the identical
formula. That pattern rules out optimizer noise and data order (both would move
train and val together) and points at the log-variance head's calibration.

Hypothesis H1
-------------
The Kendall-Gal term is `exp(-s) * huber(e) + 0.5 * s` (psi_full_pipeline.py).
Minimising over s gives `s* = ln(2 * huber)`, so pairs fit better than
huber ~= 1e-3 are driven into the -7 clamp. On val the head emits the same
confidence but the error is ~1.66x larger, so those pairs contribute
`exp(7) * huber ~= 1097 * huber` and a handful of them dominate the whole val
average -- which handful changes every epoch, hence the bouncing.

H1 was derived, never measured. This run measures it. It is deliberately cheap
(~15 min) because it can also *falsify* the hypothesis, in which case none of the
downstream ablation phases are worth building.

Usage
-----
    python phase_check_1.py                 # run training, then write the report
    python phase_check_1.py --analyze-only  # re-write the report from an existing run

Reads  runs/phase1/training_history.json
Writes PHASE1_REPORT.md
"""

import argparse
import os

import phase_common as pc

# --- Experiment definition ---------------------------------------------------
# Scale, LR horizon and paths are shared with the other phases (phase_common).
# 40 epochs here (vs 30 for the sweeps) because this phase needs curve *shape*,
# not just between-arm ordering.
RUN_ID = "phase1"
EPOCHS = 40
TARGET_REACTIONS = pc.TARGET_REACTIONS
LR_SCHEDULE_EPOCHS = pc.LR_SCHEDULE_EPOCHS
DEFAULT_DATA_DIR = pc.DEFAULT_DATA_DIR
DEFAULT_SAMPLE_CACHE = pc.DEFAULT_SAMPLE_CACHE
FULL_SCALE_REF = pc.FULL_SCALE_REF

# Decision thresholds. Stated up front so the verdict is mechanical rather than
# chosen after seeing the numbers.
PIN_CONFIRM = 0.40    # val pinned_lo at/above this => head has saturated
PIN_REJECT = 0.05     # val pinned_lo below this => H1 is wrong
RATIO_CONFIRM = 20.0  # NLL must be >= this many times more volatile than the MAE
TAIL_WINDOW = 15      # epochs averaged for the volatility metrics


def run_training(args):
    pc.check_sample_cache(args.sample_cache)
    print("Phase 1 run:")
    pc.launch(
        pc.base_command(args.run_dir, args.sample_cache, args.data_dir, EPOCHS),
        args.run_dir,
    )


def load_history(run_dir):
    history = pc.load_history(run_dir, require_logvar=True)
    if history[-1]["val_logvar"] is None:
        raise SystemExit(
            "val_logvar is null for every epoch, which means geom_uncertainty was "
            "off for this run. Phase 1 has nothing to measure -- re-run with "
            "geom_uncertainty enabled (it is the CONFIG default)."
        )
    return history


mean_abs_step = pc.mean_abs_step


def analyze(history):
    tail = history[-TAIL_WINDOW:] if len(history) > TAIL_WINDOW else history
    first, last = history[0], history[-1]

    bounce = mean_abs_step([e["val_geom"] for e in tail])
    smooth = mean_abs_step([e["val_geom_mae_A"] for e in tail])
    train_bounce = mean_abs_step([e["train_geom"] for e in tail])

    gn = last["train_grad_norm"]
    stats = {
        "epochs": len(history),
        "bounce": bounce,
        "smooth": smooth,
        # How many times more volatile the NLL is than the MAE underneath it. This
        # is the signature being reproduced: at full scale it was ~227.
        "ratio": (bounce / smooth) if smooth > 0 else float("inf"),
        "train_bounce": train_bounce,
        "gap": last["val_geom_mae_A"] / last["train_geom_mae_A"],
        "val_mae": last["val_geom_mae_A"],
        "train_mae": last["train_geom_mae_A"],
        "val_nll": last["val_geom"],
        "train_nll": last["train_geom"],
        "pin_lo_val_first": first["val_logvar"]["pinned_lo"],
        "pin_lo_val_last": last["val_logvar"]["pinned_lo"],
        "pin_lo_train_last": last["train_logvar"]["pinned_lo"],
        "pin_hi_val_last": last["val_logvar"]["pinned_hi"],
        "p50_train": last["train_logvar"]["p50"],
        "p50_val": last["val_logvar"]["p50"],
        "p1_val": last["val_logvar"]["p1"],
        "p99_val": last["val_logvar"]["p99"],
        "grad_p50": gn["p50"] if gn else None,
        "grad_p99": gn["p99"] if gn else None,
        "grad_tail": (gn["p99"] / gn["p50"]) if gn and gn["p50"] > 0 else None,
        "clip_rate": gn["clip_rate"] if gn else None,
        "skips": sum(e.get("train_nonfinite_grad_skips", 0) for e in history),
    }
    stats["pin_rising"] = stats["pin_lo_val_last"] > stats["pin_lo_val_first"]
    return stats


def verdict(s):
    """Apply the pre-registered decision rule. Returns (label, reasoning)."""
    saturated = s["pin_lo_val_last"] >= PIN_CONFIRM
    volatile = s["ratio"] >= RATIO_CONFIRM

    if saturated and volatile and s["pin_rising"]:
        return "H1 CONFIRMED", (
            f"The log-variance head has saturated: {s['pin_lo_val_last']:.1%} of val atoms "
            f"sit at the -7 clamp (up from {s['pin_lo_val_first']:.1%} at epoch 1), each "
            f"carrying the full ~1097x gradient amplification. The val NLL is "
            f"{s['ratio']:.0f}x more volatile than the val MAE beneath it, reproducing the "
            f"full-scale signature (~{FULL_SCALE_REF['ratio']:.0f}x). Clamp saturation is "
            "the mechanism. Phase 2 (clamp sweep) is justified: tightening the bound "
            "should reduce the bounce monotonically."
        )
    if s["pin_lo_val_last"] < PIN_REJECT:
        return "H1 REJECTED", (
            f"Only {s['pin_lo_val_last']:.1%} of val atoms are at the clamp, below the "
            f"{PIN_REJECT:.0%} rejection threshold. The head is NOT saturating, so the "
            "exp(-7)=1097x amplification story cannot explain the volatility. Do not "
            "build Phase 2 -- re-diagnose first. The val log-variance distribution "
            "(p1/p50/p99 below) is the place to start."
        )
    if saturated and not volatile:
        return "INCONCLUSIVE", (
            f"The head has saturated ({s['pin_lo_val_last']:.1%} at the clamp) but the "
            f"volatility signature did not reproduce at this scale ({s['ratio']:.0f}x vs "
            f"~{FULL_SCALE_REF['ratio']:.0f}x at full scale). Saturation may be necessary "
            "but not sufficient. Most likely 40 epochs is too short for the val gap to "
            "open far enough -- re-run at 80 epochs before deciding on Phase 2."
        )
    return "INCONCLUSIVE", (
        f"val pinned_lo = {s['pin_lo_val_last']:.1%} falls between the rejection "
        f"({PIN_REJECT:.0%}) and confirmation ({PIN_CONFIRM:.0%}) thresholds, and the "
        f"volatility ratio is {s['ratio']:.0f}x. The head is drifting toward the clamp "
        "without having reached it. Re-run at 80 epochs to see whether it saturates."
    )


def clip_note(s):
    """Flag clip saturation. A clip rate at or near 1.0 means `grad_clip` is not the
    outlier guard it was configured to be but an unconditional rescale of every step,
    which discards gradient magnitude and leaves `lr` as the sole step-size control.
    That is a first-class finding, not a footnote to the log-variance result."""
    rate = s["clip_rate"]
    if rate is None:
        return "No gradient norms were recorded (no optimizer steps taken)."
    if rate >= 0.99:
        return (
            f"**CLIP SATURATION — {rate:.1%} of steps were clipped.** The median gradient "
            f"norm is {s['grad_p50']:.2f} against a `grad_clip` of 1.0, so the threshold "
            f"sits {s['grad_p50']:.1f}x below the *median* rather than catching outliers. "
            "Every update is therefore a fixed-norm direction with its magnitude "
            "discarded, removing the \"gradients are shrinking, we are near a minimum\" "
            "signal and leaving `lr` as the sole determinant of step size — a coherent "
            "mechanism for larger LR converging worse. Genuine outliers do exist "
            f"(p99 {s['grad_p99']:.0f}, {s['skips']} non-finite skips), so some clipping is "
            "load-bearing. Phase 2 sweeps the threshold to separate the two."
        )
    if rate >= 0.25:
        return (
            f"{rate:.1%} of steps were clipped — high enough that `grad_clip` is shaping "
            "training rather than only guarding against outliers, but not saturated. "
            "Worth a threshold sweep."
        )
    return (
        f"{rate:.1%} of steps were clipped, consistent with `grad_clip` acting as an "
        "outlier guard as intended. Not a lever on the LR question."
    )


def epoch_table(history):
    """Sampled epoch rows -- enough to read the trend without dumping every epoch."""
    step = max(1, len(history) // 10)
    picked = [e for i, e in enumerate(history) if i % step == 0 or e is history[-1]]
    lines = [
        "| epoch | val NLL | val MAE (A) | train pin_lo | val pin_lo | val logvar p50 | grad p99/p50 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for e in picked:
        tl, vl, gn = e["train_logvar"], e["val_logvar"], e["train_grad_norm"]
        tail = f"{gn['p99'] / gn['p50']:.1f}" if gn and gn["p50"] > 0 else "n/a"
        lines.append(
            f"| {e['epoch']} | {e['val_geom']:.4f} | {e['val_geom_mae_A']:.4f} | "
            f"{tl['pinned_lo']:.1%} | {vl['pinned_lo']:.1%} | {vl['p50']:.2f} | {tail} |"
        )
    return "\n".join(lines)


def write_report(history, stats, path):
    label, reasoning = verdict(stats)
    grad_tail = f"{stats['grad_tail']:.1f}x" if stats["grad_tail"] else "n/a"
    clip = f"{stats['clip_rate']:.1%}" if stats["clip_rate"] is not None else "n/a"

    report = f"""# Phase 1 — Log-variance observation run

**Verdict: {label}**

{reasoning}

## What was run

A single {EPOCHS}-epoch run on {TARGET_REACTIONS:,} reactions, cosine horizon pinned to
{LR_SCHEDULE_EPOCHS} epochs so the LR stays near peak throughout (an uncompressed
schedule is required to observe high-LR behaviour in a short run). Early stopping
disabled. Completed {stats['epochs']} epochs.

## Scale caveat

Absolute values at {TARGET_REACTIONS:,} reactions do not transfer to the 40k production
run — the train/val gap opens faster and wider with less data. The full-scale column
below is for orientation only; the verdict rests on the *pattern* (saturation present
or absent, volatility ratio large or small), not on matching numbers.

## Log-variance distribution — the measurement

| quantity | train | val |
|---|---:|---:|
| fraction pinned at -7 clamp | {stats['pin_lo_train_last']:.1%} | **{stats['pin_lo_val_last']:.1%}** |
| fraction pinned at +7 clamp | — | {stats['pin_hi_val_last']:.1%} |
| logvar p1 | — | {stats['p1_val']:.2f} |
| logvar p50 | {stats['p50_train']:.2f} | {stats['p50_val']:.2f} |
| logvar p99 | — | {stats['p99_val']:.2f} |

val pinned_lo moved {stats['pin_lo_val_first']:.1%} → {stats['pin_lo_val_last']:.1%}
across the run ({'rising' if stats['pin_rising'] else 'not rising'}).

A pinned atom contributes `exp(7) ≈ 1097x` amplification to its pairs' gradient and
loss. The pinned fraction is therefore the single number that determines whether the
NLL is a well-behaved objective or a lever arm on a few outliers.

## Volatility — did the signature reproduce?

| metric | this run | full-scale ref (ep 100-147) |
|---|---:|---:|
| BOUNCE — mean \\|Δ val NLL\\|/epoch | {stats['bounce']:.4f} | {FULL_SCALE_REF['bounce']:.4f} |
| SMOOTH — mean \\|Δ val MAE\\|/epoch | {stats['smooth']:.5f} | {FULL_SCALE_REF['smooth']:.5f} |
| **RATIO — BOUNCE / SMOOTH** | **{stats['ratio']:.0f}x** | {FULL_SCALE_REF['ratio']:.0f}x |
| mean \\|Δ train NLL\\|/epoch | {stats['train_bounce']:.4f} | 0.0048 |
| GAP — val MAE / train MAE | {stats['gap']:.2f} | {FULL_SCALE_REF['gap']:.2f} |

Both volatility figures are averaged over the last {TAIL_WINDOW} epochs. The train-NLL
row is the control: if train is smooth while val bounces, the instability is
generalization, not optimization.

## Gradient scale

| quantity | value |
|---|---:|
| pre-clip grad norm p50 | {stats['grad_p50']:.3f} |
| pre-clip grad norm p99 | {stats['grad_p99']:.3f} |
| **tail ratio p99/p50** | **{grad_tail}** |
| steps where the clip bit | {clip} |
| non-finite grad skips (all epochs) | {stats['skips']} |

Recorded on steps that actually took an update; skipped steps contribute no norm.

{clip_note(stats)}

## Trend

{epoch_table(history)}

## What this does and does not settle

Settles: whether the log-variance head saturates at the clamp, and whether the
val-NLL volatility signature reproduces at reduced scale.

Does not settle: whether saturation *causes* the volatility (Phase 2, clamp sweep)
or whether it drives the LR sensitivity (Phase 3, LR × uncertainty factorial).
Those need ablations; this run is observation only.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="RGD1 dataset directory")
    parser.add_argument("--run-dir", default=os.path.join(pc.RUNS_DIR, RUN_ID), help="Output directory")
    parser.add_argument("--sample-cache", default=DEFAULT_SAMPLE_CACHE,
                        help="Existing sample cache to slice 4k reactions from")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip training; rebuild the report from an existing run")
    parser.add_argument("--report", default=os.path.join(pc.PHASES_DIR, "PHASE1_REPORT.md"),
                        help="Report output path")
    args = parser.parse_args()

    if not args.analyze_only:
        run_training(args)

    history = load_history(args.run_dir)
    stats = analyze(history)
    label = write_report(history, stats, args.report)

    print(f"\n{'=' * 60}")
    print(f" PHASE 1 VERDICT: {label}")
    print(f"{'=' * 60}")
    print(f" val pinned at -7 clamp : {stats['pin_lo_val_last']:.1%} "
          f"(epoch 1: {stats['pin_lo_val_first']:.1%})")
    print(f" NLL/MAE volatility     : {stats['ratio']:.0f}x  (full scale ~227x)")
    print(f" val/train MAE gap      : {stats['gap']:.2f}")
    print(f"\n Report written to {args.report}")


if __name__ == "__main__":
    main()
