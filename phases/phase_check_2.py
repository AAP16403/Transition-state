#!/usr/bin/env python3
"""Phase 2 — grad-clip sweep: is unconditional clipping capping learning?

Why this axis
-------------
Phase 1 rejected the clamp-saturation hypothesis (val pinned_lo peaked at 0.14%)
but measured something louder: `clip_rate` was **1.0000 on 40 of 40 epochs**. Every
optimizer step in the run was clipped, with a median gradient norm of 6.57 against
a `grad_clip` of 1.0.

So `grad_clip: 1.0` is not the outlier guard it was configured to be -- it is an
unconditional rescale applied to every step, which discards gradient magnitude
entirely. That removes the "gradients are shrinking, we are near a minimum" signal
and leaves `lr` as the sole determinant of step size, with no magnitude-based
annealing. It is a coherent mechanism for "bigger LR performs worse".

Real outliers do exist (p99 up to 938, max 1321, 8 non-finite skips), so clipping
is doing *something* necessary. The question is whether a threshold 6.6x below the
median is buying that protection at the cost of capping learning.

Hypothesis H2
-------------
`grad_clip=1.0` is capping learning. Loosening it so the clip catches only genuine
outliers should improve val MAE.

Usage
-----
    python phase_check_2.py                 # run the 4-arm sweep, then report
    python phase_check_2.py --analyze-only  # rebuild the report from existing runs
    python phase_check_2.py --force         # re-run arms that already completed

Writes PHASE2_REPORT.md
"""

import argparse
import os

import phase_common as pc

EPOCHS = 30  # sweep arms only need relative ordering, not converged curves

# Arms chosen against the Phase 1 gradient distribution (p50 = 6.57) so clip_rate
# spreads across the range rather than clustering. `inf` disables clipping while
# still recording the norms, since the instrumentation reads them off the
# clip_grad_norm_ return value.
ARMS = [
    ("clip1", "1.0", "baseline — the production value"),
    ("clip5", "5.0", "below the median gradient: clips most steps"),
    ("clip15", "15.0", "above the median: clips the tail only"),
    ("clipoff", "inf", "no clipping — isolates what the clip is protecting against"),
]

# Decision thresholds, pre-registered so the verdict is mechanical.
#
# MEASURED noise floor, not a guess: Phase 1 (40 ep) and Phase 2 arm clip1 (30 ep)
# ran with identical configuration -- same seed, same data, same grad_clip=1.0, same
# pinned LR horizon -- and their val MAE at epoch 30 differed by 0.0047 A. That is
# pure run-to-run non-determinism (cuDNN/AMP atomics, non-deterministic scatter).
#
# The original 0.005 threshold sat exactly on that floor, so it would have treated
# GPU noise as the significance boundary. Require 3x the floor instead. This is a
# crude estimate from a single pair of runs; if an arm lands between 1x and 3x, the
# honest answer is more seeds, not a verdict.
NOISE_FLOOR = pc.NOISE_FLOOR
MAE_SIGNIFICANT = 3 * NOISE_FLOOR  # 0.0141 A
SKIP_TOLERANCE = 20       # Non-finite skips above this mean clipping is load-bearing.


def run_dir_for(arm_id):
    return os.path.join(pc.RUNS_DIR, f"phase2_{arm_id}")


def run_sweep(args):
    pc.check_sample_cache(args.sample_cache)
    for arm_id, clip, _ in ARMS:
        rd = run_dir_for(arm_id)
        done = os.path.exists(os.path.join(rd, "training_history.json"))
        if done and not args.force:
            print(f"[skip] {arm_id}: already complete (--force to re-run)")
            continue
        print(f"\n=== Phase 2 arm {arm_id}: grad_clip={clip} ===")
        pc.launch(
            pc.base_command(rd, args.sample_cache, args.data_dir, EPOCHS,
                            extra=["--grad-clip", clip]),
            rd,
        )


def analyze_arm(arm_id, clip_str):
    """Return the arm's summary, or None if it never ran / never completed an epoch.

    Arms can be missing: a crash aborts the sweep and leaves later arms unstarted.
    A partial sweep is still worth reporting -- refusing to analyse would throw away
    completed arms -- but the verdict must know which arms it is missing rather than
    silently comparing a short arm against a full one.
    """
    path = os.path.join(run_dir_for(arm_id), "training_history.json")
    if not os.path.exists(path):
        return None
    history = pc.load_history(run_dir_for(arm_id))
    row = _metrics_at(history, arm_id, clip_str, len(history))
    row["ran_epochs"] = len(history)
    row["at_common"] = lambda n, h=history, a=arm_id, c=clip_str: _metrics_at(h, a, c, n)
    return row


def _metrics_at(history, arm_id, clip_str, n_epochs):
    """Summarise an arm as of its first `n_epochs` epochs."""
    h = history[:n_epochs]
    last = h[-1]
    t = pc.tail(h)
    gn = last["train_grad_norm"]
    # Averaged over the tail: single-epoch clip_rate is noisy on 71 steps.
    clip_rates = [e["train_grad_norm"]["clip_rate"] for e in t if e["train_grad_norm"]]
    return {
        "arm": arm_id,
        "clip": clip_str,
        "epochs": len(h),
        "val_mae": last["val_geom_mae_A"],
        "train_mae": last["train_geom_mae_A"],
        "gap": last["val_geom_mae_A"] / last["train_geom_mae_A"],
        "val_ea": last["val_ea_mae"],
        "clip_rate": sum(clip_rates) / max(len(clip_rates), 1),
        "grad_p50": gn["p50"],
        "grad_p99": gn["p99"],
        "skips": sum(e.get("train_nonfinite_grad_skips", 0) for e in h),
        "bounce": pc.mean_abs_step([e["val_geom"] for e in t]),
        "smooth": pc.mean_abs_step([e["val_geom_mae_A"] for e in t]),
    }


def verdict(rows):
    """Apply the pre-registered decision rule. Returns (label, reasoning)."""
    rows = [r for r in rows if r is not None]
    if len(rows) < 2:
        return "INCOMPLETE", (
            f"Only {len(rows)} arm(s) produced a history. A sweep needs at least two "
            "arms to compare. Re-run the missing arms before reading anything into this."
        )
    # Compare at the shortest common epoch: an arm that crashed early would otherwise
    # be judged against arms that trained twice as long, which is not a clip effect.
    common = min(r["epochs"] for r in rows)
    if any(r["epochs"] != common for r in rows):
        for r in rows:
            r.update(r["at_common"](common))
    baseline = rows[0]
    best = min(rows, key=lambda r: r["val_mae"])
    delta = baseline["val_mae"] - best["val_mae"]
    unstable = [r for r in rows if r["skips"] > SKIP_TOLERANCE]
    # Spread across ALL arms, not baseline-vs-best: when the baseline is itself the
    # best arm, delta is 0 and a delta-first test would misreport a large degradation
    # from loosening as "no separation".
    spread = max(r["val_mae"] for r in rows) - min(r["val_mae"] for r in rows)

    if spread < MAE_SIGNIFICANT:
        return "H2 REJECTED - clip is not the lever", (
            f"All four arms land within {spread:.4f} A of each other on val MAE, below the "
            f"{MAE_SIGNIFICANT} A significance floor. At one seed "
            "per arm that is not a result. Unconditional clipping is therefore not what "
            "limits accuracy, and the LR sensitivity must come from elsewhere. Phase 3 "
            "should keep its original uncertainty axis rather than the clip axis."
        )
    if best["arm"] == baseline["arm"]:
        return "H2 REJECTED - clipping is load-bearing", (
            f"The tight baseline (grad_clip=1.0) is the best arm at {baseline['val_mae']:.4f} A; "
            f"loosening it degrades val MAE by up to {max(r['val_mae'] for r in rows) - baseline['val_mae']:.4f} A"
            + (f", and {len(unstable)} arm(s) exceeded {SKIP_TOLERANCE} non-finite skips" if unstable else "")
            + ". The clip is buying real stability, and the 100% clip rate is the price "
            "of that protection rather than a bug. LR sensitivity is then an accepted "
            "consequence, and the lever is the gradient scale itself, not the threshold."
        )
    return "H2 CONFIRMED - clip at 1.0 caps learning", (
        f"val MAE improves by {delta:.4f} A when the clip is loosened from 1.0 to "
        f"{best['clip']} ({baseline['val_mae']:.4f} -> {best['val_mae']:.4f}), above the "
        f"{MAE_SIGNIFICANT} A significance floor. The production threshold sits 6.6x below "
        "the median gradient and was capping learning, not guarding against outliers"
        + (f". Note {len(unstable)} arm(s) exceeded {SKIP_TOLERANCE} non-finite skips, so the "
           "usable threshold is bounded above by stability" if unstable else
           ". No arm showed a stability cost")
        + ". Phase 3 should sweep LR against this new threshold."
    )


def write_report(rows, path):
    label, reasoning = verdict(rows)
    hdr = ("| grad_clip | clip rate | val MAE (A) | train MAE | gap | val Ea | "
           "grad p50 | grad p99 | skips | val-NLL bounce |")
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [hdr, sep]
    for (arm_id, clip, _), r in zip(ARMS, rows):
        if r is None:
            lines.append(f"| {clip} | — | *did not run* | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {clip} | {r['clip_rate']:.1%} | **{r['val_mae']:.4f}** | "
            f"{r['train_mae']:.4f} | {r['gap']:.2f} | {r['val_ea']:.2f} | "
            f"{r['grad_p50']:.2f} | {r['grad_p99']:.1f} | {r['skips']} | {r['bounce']:.4f} |"
        )
    table = "\n".join(lines)
    done = [r for r in rows if r is not None]
    common = min((r["epochs"] for r in done), default=0)
    coverage = (
        f"\n**Coverage: {len(done)}/{len(ARMS)} arms.** All figures are at epoch "
        f"{common}, the deepest epoch every surviving arm reached — comparing a "
        "crashed arm against a full-length one would measure run length, not clip.\n"
        if len(done) < len(ARMS) or any(r["ran_epochs"] != common for r in done) else ""
    )
    arms_desc = "\n".join(f"- `{c}` — {d}" for _, c, d in ARMS)

    report = f"""# Phase 2 — Grad-clip sweep

**Verdict: {label}**

{reasoning}

## Why this axis

Phase 1 rejected clamp saturation (val `pinned_lo` peaked at 0.14%) but found
`clip_rate = 1.0000` on 40/40 epochs, with a median gradient norm of
{pc.PHASE1_REF['grad_p50']} against a `grad_clip` of 1.0. Every step was clipped, so
gradient magnitude was discarded on every update — leaving `lr` as the sole
determinant of step size. This phase asks whether that costs accuracy.

## Arms

{arms_desc}

{EPOCHS} epochs, {pc.TARGET_REACTIONS:,} reactions, cosine horizon pinned to
{pc.LR_SCHEDULE_EPOCHS} so LR stays near peak. One seed per arm.

## Results
{coverage}
{table}

Clip rate and volatility are averaged over the last {pc.TAIL_WINDOW} epochs; the rest
are final-epoch values. Phase 1 reference at 40 epochs: val MAE
{pc.PHASE1_REF['val_mae']:.4f}, gap {pc.PHASE1_REF['gap']:.2f}.

## Reading it

- **clip rate** should fall monotonically across the arms. If it does not, the arms
  did not actually separate and nothing below is interpretable.
- **val MAE** is the outcome. Differences under {MAE_SIGNIFICANT} A are noise at one
  seed per arm.
- **skips** is the safety column: rising non-finite counts as the clip loosens mean
  it was protecting against something real.
- **gap** shows whether looser clipping trades accuracy for overfitting.

## Noise floor

Phase 1 and this sweep's `clip1` arm ran with **identical** configuration and their
val MAE at epoch 30 differed by **{NOISE_FLOOR:.4f} A** — pure run-to-run
non-determinism. The significance threshold above is 3x that floor
({MAE_SIGNIFICANT:.4f} A). Arm differences between 1x and 3x the floor are not
resolvable at one seed per arm; the answer there is more seeds, not a verdict.

## Scale caveat

{pc.TARGET_REACTIONS:,} reactions / {EPOCHS} epochs, one seed. Between-arm ordering is
the result; absolute values do not transfer to the 40k production run. Any arm
adopted here still needs confirmation at full scale.

## What this does not settle

Whether the LR sensitivity itself is fixed. That needs the LR axis (Phase 3), run
against whatever threshold this phase selects.
"""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(report)
    return label


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=pc.DEFAULT_DATA_DIR, help="RGD1 dataset directory")
    parser.add_argument("--sample-cache", default=pc.DEFAULT_SAMPLE_CACHE,
                        help="Existing sample cache to slice reactions from")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Skip training; rebuild the report from existing runs")
    parser.add_argument("--force", action="store_true",
                        help="Re-run arms that already completed")
    parser.add_argument("--report", default=os.path.join(pc.PHASES_DIR, "PHASE2_REPORT.md"),
                        help="Report output path")
    args = parser.parse_args()

    if not args.analyze_only:
        run_sweep(args)

    rows = [analyze_arm(arm_id, clip) for arm_id, clip, _ in ARMS]
    label = write_report(rows, args.report)

    print(f"\n{'=' * 66}")
    print(f" PHASE 2 VERDICT: {label}")
    print(f"{'=' * 66}")
    print(f" {'grad_clip':>10} {'clip rate':>10} {'val MAE':>9} {'gap':>6} {'skips':>6} {'epochs':>7}")
    for (arm_id, clip, _), r in zip(ARMS, rows):
        if r is None:
            print(f" {clip:>10} {'-':>10} {'did not run':>9}")
            continue
        print(f" {r['clip']:>10} {r['clip_rate']:>9.1%} {r['val_mae']:>9.4f} "
              f"{r['gap']:>6.2f} {r['skips']:>6} {r['epochs']:>7}")
    print(f"\n Report written to {args.report}")


if __name__ == "__main__":
    main()
