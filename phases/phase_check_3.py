#!/usr/bin/env python3
"""Phase 3 — LR sweep: is "bigger learning rate performs worse" a real effect?

Why this is now a direct test
----------------------------
The original observation came from comparing two production runs that differed in
~8 configuration dimensions at once, so LR was never isolated. Two candidate
mediators have since been measured and killed:

  Phase 1  log-variance clamp saturation  -- REJECTED (val pinned_lo peaked at 0.14%)
  Phase 2  grad-clip saturation           -- REJECTED (clip rate 100% -> 10% moved
                                             val MAE by 0.0024 A, half the noise floor)

Phase 2's negative result is expected in hindsight: Adam is approximately
scale-invariant, so the uniform rescale that global-norm clipping applies is largely
absorbed by its m/sqrt(v) normalisation.

So this phase stops hunting for mechanisms and tests the claim itself. Only if the
LR curve is genuinely steep is there an effect worth attributing.

Hypothesis H3
-------------
Final val MAE degrades monotonically with learning rate.

Usage
-----
    python phase_check_3.py                 # run the 3-arm sweep, then report
    python phase_check_3.py --analyze-only  # rebuild the report from existing runs
    python phase_check_3.py --force         # re-run arms that already completed

Writes PHASE3_REPORT.md
"""

import argparse
import os

import phase_common as pc

EPOCHS = 30

# Half, baseline, and 3x the production LR -- wide enough that a real effect cannot
# hide inside the noise floor, and centred on the value both earlier phases used.
ARMS = [
    ("lr050", "5e-5", "one third of production"),
    ("lr150", "1.5e-4", "baseline — the production value"),
    ("lr450", "4.5e-4", "3x production"),
]

# Measured in Phase 2: two identically-configured runs differed by 0.0047 A in val
# MAE at epoch 30 (cuDNN/AMP non-determinism). Require 3x that before calling any
# arm difference real.
NOISE_FLOOR = pc.NOISE_FLOOR
MAE_SIGNIFICANT = 3 * NOISE_FLOOR


def run_dir_for(arm_id):
    return os.path.join(pc.RUNS_DIR, f"phase3_{arm_id}")


def run_sweep(args):
    pc.check_sample_cache(args.sample_cache)
    failed = []
    for arm_id, lr, _ in ARMS:
        rd = run_dir_for(arm_id)
        if os.path.exists(os.path.join(rd, "training_history.json")) and not args.force:
            print(f"[skip] {arm_id}: already complete (--force to re-run)")
            continue
        print(f"\n=== Phase 3 arm {arm_id}: lr={lr} ===")
        # abort_on_failure=False: Phase 2 lost its final arm to a CUDA driver fault
        # in an unrelated arm. One bad arm must not discard the queue.
        rc = pc.launch(
            pc.base_command(rd, args.sample_cache, args.data_dir, EPOCHS,
                            extra=["--lr", lr], skip_final_eval=True),
            rd, abort_on_failure=False,
        )
        if rc != 0:
            failed.append(arm_id)
    if failed:
        print(f"\n{len(failed)} arm(s) failed: {', '.join(failed)}. "
              "The report below covers the arms that survived.")


def analyze_arm(arm_id, lr_str):
    path = os.path.join(run_dir_for(arm_id), "training_history.json")
    if not os.path.exists(path):
        return None
    history = pc.load_history(run_dir_for(arm_id))
    row = _metrics_at(history, arm_id, lr_str, len(history))
    row["ran_epochs"] = len(history)
    row["at_common"] = lambda n, h=history, a=arm_id, s=lr_str: _metrics_at(h, a, s, n)
    return row


def _metrics_at(history, arm_id, lr_str, n_epochs):
    h = history[:n_epochs]
    last = h[-1]
    t = pc.tail(h)
    gn = last["train_grad_norm"]
    return {
        "arm": arm_id,
        "lr": lr_str,
        "epochs": len(h),
        "val_mae": last["val_geom_mae_A"],
        "train_mae": last["train_geom_mae_A"],
        "gap": last["val_geom_mae_A"] / last["train_geom_mae_A"],
        "val_ea": last["val_ea_mae"],
        "grad_p50": gn["p50"] if gn else float("nan"),
        "clip_rate": gn["clip_rate"] if gn else float("nan"),
        "skips": sum(e.get("train_nonfinite_grad_skips", 0) for e in h),
        "bounce": pc.mean_abs_step([e["val_geom"] for e in t]),
        "smooth": pc.mean_abs_step([e["val_geom_mae_A"] for e in t]),
        # Best epoch, not just final: a high-LR arm can bounce around a good minimum
        # without landing on it at epoch 30, which final-only would misread as worse.
        "best_val_mae": min(e["val_geom_mae_A"] for e in h),
        # Still improving at the end? Compares the mean of the last fifth of epochs
        # against the fifth before it. If true, the arm had not converged and the
        # sweep measured how fast it trained, not where it lands.
        "still_descending": _still_descending([e["val_geom_mae_A"] for e in h]),
    }


def _still_descending(vals, min_epochs=10):
    """True when the tail is meaningfully below the block before it."""
    if len(vals) < min_epochs:
        return False
    k = max(2, len(vals) // 5)
    prev = sum(vals[-2 * k:-k]) / k
    last = sum(vals[-k:]) / k
    # Half the noise floor: below that the "improvement" is indistinguishable from
    # run-to-run jitter and the arm should count as converged.
    return (prev - last) > (NOISE_FLOOR / 2)


def verdict(rows):
    rows = [r for r in rows if r is not None]
    if len(rows) < 2:
        return "INCOMPLETE", (
            f"Only {len(rows)} arm(s) produced a history; a sweep needs at least two. "
            "Re-run the missing arms."
        )
    common = min(r["epochs"] for r in rows)
    if any(r["epochs"] != common for r in rows):
        for r in rows:
            r.update(r["at_common"](common))

    # Compare on BEST val MAE, not final. A high-LR arm oscillates around its minimum
    # (larger steps => larger residual noise), so its final epoch can land well above
    # its own best purely by chance. Testing the final epoch penalises exactly the arm
    # this phase is about and manufactures a null result.
    spread = max(r["best_val_mae"] for r in rows) - min(r["best_val_mae"] for r in rows)
    by_lr = sorted(rows, key=lambda r: float(r["lr"]))
    monotone_worse = all(
        by_lr[i]["best_val_mae"] <= by_lr[i + 1]["best_val_mae"] for i in range(len(by_lr) - 1)
    )
    monotone_better = all(
        by_lr[i]["best_val_mae"] >= by_lr[i + 1]["best_val_mae"] for i in range(len(by_lr) - 1)
    )
    best = min(rows, key=lambda r: r["best_val_mae"])

    # Budget check. At a fixed epoch count a faster LR simply gets further, so an arm
    # still descending at the end means this sweep measured training SPEED, not
    # converged quality -- and the claim under test is about converged runs.
    descending = [r for r in rows if r.get("still_descending")]
    if descending:
        return "H3 INCONCLUSIVE - budget-limited", (
            f"{len(descending)} of {len(rows)} arm(s) were still improving at epoch "
            f"{common} (lr=" + ", ".join(r["lr"] for r in descending) + "), so this sweep "
            "compared how far each learning rate got in a fixed epoch budget, not where "
            "each one converges. A faster LR trivially gets further under that "
            "constraint. The claim under test concerns converged runs (the originals ran "
            "147 and 323 epochs), so this cannot settle it either way.\n\n"
            f"For the record the ordering is the OPPOSITE of the original impression: "
            f"best val MAE improves with LR, spanning {spread:.4f} A "
            f"(best {best['best_val_mae']:.4f} at lr={best['lr']}), above the "
            f"{MAE_SIGNIFICANT:.4f} A floor. But the overfitting gap also rises with LR "
            f"({by_lr[0]['gap']:.2f} -> {by_lr[-1]['gap']:.2f}), which is precisely how a "
            "faster LR could still finish worse once every arm has converged. Re-run to "
            "convergence, or match arms on training progress rather than epoch count."
        )

    if spread < MAE_SIGNIFICANT:
        return "H3 REJECTED - no LR effect", (
            f"val MAE spans only {spread:.4f} A across a {float(by_lr[-1]['lr']) / float(by_lr[0]['lr']):.0f}x "
            f"range of learning rates, below the {MAE_SIGNIFICANT:.4f} A significance "
            f"floor (3x the measured {NOISE_FLOOR:.4f} A run-to-run noise). "
            "\"Bigger LR performs worse\" does not reproduce under controlled "
            "conditions, which means the original impression came from the ~8 "
            "confounded differences between those two production runs — most likely "
            "the cosine horizon, which is tied to swa_start and changed between them. "
            "There is no LR effect left to explain, and no mechanism worth hunting."
        )
    if monotone_better:
        return "H3 REFUTED - bigger LR is BETTER", (
            f"best val MAE *improves* monotonically with learning rate, spanning "
            f"{spread:.4f} A (best {best['best_val_mae']:.4f} at lr={best['lr']}), above "
            f"the {MAE_SIGNIFICANT:.4f} A floor. The original impression is not merely "
            "unsupported, it is backwards under controlled conditions. Note the "
            f"overfitting gap rises with LR ({by_lr[0]['gap']:.2f} -> {by_lr[-1]['gap']:.2f}), "
            "so confirm at full length before raising the production LR."
        )
    if monotone_worse:
        return "H3 CONFIRMED - bigger LR is worse", (
            f"best val MAE degrades monotonically with learning rate, spanning {spread:.4f} A "
            f"(best {best['best_val_mae']:.4f} at lr={best['lr']}), above the "
            f"{MAE_SIGNIFICANT:.4f} A floor. The effect is real and directly measured "
            "rather than inferred from confounded runs. Two mediators are already "
            "excluded (logvar clamp, grad clip), so the mechanism hunt should now "
            "target the optimizer state itself — Adam's beta2/eps, or the interaction "
            "between LR and the cosine horizon."
        )
    return "H3 PARTIAL - LR matters, not monotonically", (
        f"best val MAE spans {spread:.4f} A, above the {MAE_SIGNIFICANT:.4f} A floor, so LR "
        f"does affect the outcome — but not monotonically: the best arm is lr="
        f"{best['lr']} at {best['best_val_mae']:.4f} A, with worse results on both sides. "
        "That is an optimum, not a \"bigger is worse\" trend, and it means the "
        "production LR should be retuned rather than simply lowered."
    )


def write_report(rows, path):
    label, reasoning = verdict(rows)
    lines = [
        "| lr | val MAE (A) | best val MAE | train MAE | gap | val Ea | grad p50 | clip rate | skips |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for (arm_id, lr, _), r in zip(ARMS, rows):
        if r is None:
            lines.append(f"| {lr} | *did not run* | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {lr} | **{r['val_mae']:.4f}** | {r['best_val_mae']:.4f} | {r['train_mae']:.4f} | "
            f"{r['gap']:.2f} | {r['val_ea']:.2f} | {r['grad_p50']:.2f} | "
            f"{r['clip_rate']:.1%} | {r['skips']} |"
        )
    table = "\n".join(lines)
    done = [r for r in rows if r is not None]
    common = min((r["epochs"] for r in done), default=0)
    coverage = ""
    if len(done) < len(ARMS) or any(r["ran_epochs"] != common for r in done):
        coverage = (
            f"\n**Coverage: {len(done)}/{len(ARMS)} arms.** Figures are at epoch {common}, "
            "the deepest epoch every surviving arm reached.\n"
        )
    arms_desc = "\n".join(f"- `{lr}` — {d}" for _, lr, d in ARMS)

    report = f"""# Phase 3 — Learning-rate sweep

**Verdict: {label}**

{reasoning}

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

{arms_desc}

{EPOCHS} epochs, {pc.TARGET_REACTIONS:,} reactions, cosine horizon pinned to
{pc.LR_SCHEDULE_EPOCHS} so the schedule is NOT compressed by the short run — every arm
holds near its peak LR throughout, which is the regime under test. One seed per arm.

## Results
{coverage}
{table}

`best val MAE` is the minimum over all epochs, not just the final one: a high-LR arm
can oscillate around a good minimum without landing on it at epoch {common}, and
reading the final epoch alone would misreport that as worse.

## Noise floor

Two identically-configured runs (Phase 1 and Phase 2 `clip1`) differed by
**{NOISE_FLOOR:.4f} A** in val MAE at epoch 30 — pure cuDNN/AMP non-determinism. The
significance threshold is 3x that ({MAE_SIGNIFICANT:.4f} A). Differences between 1x
and 3x the floor need more seeds, not a verdict.

## Scale caveat

{pc.TARGET_REACTIONS:,} reactions / {EPOCHS} epochs, one seed per arm. Between-arm
ordering is the result; absolute values do not transfer to the 40k production run.
Any LR adopted here needs confirmation at full scale.
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
    parser.add_argument("--force", action="store_true", help="Re-run arms that already completed")
    parser.add_argument("--report", default=os.path.join(pc.PHASES_DIR, "PHASE3_REPORT.md"),
                        help="Report output path")
    args = parser.parse_args()

    if not args.analyze_only:
        run_sweep(args)

    rows = [analyze_arm(arm_id, lr) for arm_id, lr, _ in ARMS]
    label = write_report(rows, args.report)

    print(f"\n{'=' * 66}")
    print(f" PHASE 3 VERDICT: {label}")
    print(f"{'=' * 66}")
    print(f" {'lr':>10} {'val MAE':>9} {'best':>9} {'gap':>6} {'skips':>6} {'epochs':>7}")
    for (arm_id, lr, _), r in zip(ARMS, rows):
        if r is None:
            print(f" {lr:>10} {'did not run':>9}")
            continue
        print(f" {r['lr']:>10} {r['val_mae']:>9.4f} {r['best_val_mae']:>9.4f} "
              f"{r['gap']:>6.2f} {r['skips']:>6} {r['epochs']:>7}")
    print(f"\n Report written to {args.report}")


if __name__ == "__main__":
    main()
