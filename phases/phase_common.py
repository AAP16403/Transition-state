"""Shared plumbing for the phase-check diagnostic scripts.

Path anchoring, subprocess launching, history loading and the volatility metrics
live here because every phase parses the identical `training_history.json` schema
and must agree on what BOUNCE/SMOOTH/GAP mean -- duplicating those definitions per
phase is how they silently drift apart. Each phase script still runs standalone.
"""

import json
import os
import subprocess
import sys

# Anchored to this file, not the cwd, so the scripts work when double-clicked, run
# from the repo root, or run from inside phases/.
PHASES_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(PHASES_DIR)
PIPELINE = os.path.join(REPO_ROOT, "psi_full_pipeline.py")
RUNS_DIR = os.path.join(PHASES_DIR, "runs")

DEFAULT_DATA_DIR = "/mnt/d/Transition state/RGD1_Dataset"
DEFAULT_SAMPLE_CACHE = os.path.join(REPO_ROOT, "samples_cache_rgd1.pkl")

# Shared experiment scale. 4k reactions (not smaller): val volatility is a quantity
# under study, and a smaller val set would inflate it for trivial sampling reasons.
TARGET_REACTIONS = 4000
# Production cosine horizon (= swa_start). Pinned so short runs do NOT compress the
# schedule -- otherwise a 30-epoch run spends its final epochs near min_lr and cannot
# show behaviour that only appears at sustained high LR.
LR_SCHEDULE_EPOCHS = 450

# Reference values from the full 40k run, epochs 100-147, and from Phase 1 (4k/40ep).
# Absolute numbers do not transfer across scale -- orientation only, never pass/fail.
FULL_SCALE_REF = {"bounce": 0.186, "smooth": 0.00082, "ratio": 227.0, "gap": 1.66}
PHASE1_REF = {"clip_rate": 1.0, "grad_p50": 6.57, "val_mae": 0.2608, "gap": 1.23}

# MEASURED, not assumed: Phase 1 (40 ep) and Phase 2 arm clip1 (30 ep) ran with
# identical configuration -- same seed, data, grad_clip and pinned LR horizon -- and
# their val MAE at epoch 30 differed by this much. Pure cuDNN/AMP non-determinism.
# Any phase comparing arms must treat differences below ~3x this as unresolvable.
NOISE_FLOOR = 0.0047

TAIL_WINDOW = 15  # epochs averaged for the volatility metrics


def base_command(run_dir, sample_cache, data_dir, epochs, extra=None, skip_final_eval=False):
    """The invariant part of every phase run. `extra` carries the phase's own axis."""
    cmd = [
        sys.executable, PIPELINE, "train",
        "--data-dir", data_dir,
        "--device", "cuda",
        "--num-workers", "2",
        "--target-reactions", str(TARGET_REACTIONS),
        "--epochs", str(epochs),
        "--lr-schedule-epochs", str(LR_SCHEDULE_EPOCHS),
        # Early stopping off: every epoch must be observed, and the selection metric
        # is not what these phases measure.
        "--patience", "9999",
        # Inductor warmup is pure overhead on runs this short.
        "--no-compile",
        "--save-dir", run_dir,
        # MUST be explicit. The cache path defaults to save_dir/samples_cache_rgd1.pkl,
        # so a fresh --save-dir would not find the existing 3.9 GB cache and would
        # rebuild it from scratch (hours, and an OOM risk at 12 GB RAM).
        "--sample-cache-path", sample_cache,
    ]
    if skip_final_eval:
        # Measured on Phase 2: the eval + dashboard + all-val geometry atlas tail
        # costs ~18 min per arm and peaks near the 12 GB host limit, producing
        # 130 MB of output no phase report reads.
        cmd.append("--skip-final-eval")
    return cmd + list(extra or [])


def check_sample_cache(sample_cache):
    """Fail before launching rather than letting the pipeline start a multi-hour
    full rebuild of the 3.9 GB cache. A missing cache is a setup error."""
    if not os.path.exists(sample_cache):
        raise SystemExit(
            f"Sample cache not found at {sample_cache!r}.\n"
            "The phase runs slice their reactions out of the existing cache. Without "
            "it the pipeline would rebuild from scratch. Pass --sample-cache with the "
            "path to your samples_cache_rgd1.pkl."
        )


def launch(cmd, run_dir, abort_on_failure=True):
    """Run one arm. Returns the exit code.

    Sweeps pass abort_on_failure=False: Phase 2 lost its last arm because a CUDA
    driver fault in arm 3 raised and killed the whole sweep, throwing away an hour
    of queued work for a failure unrelated to the remaining arms. A failed arm is
    reported loudly and the sweep continues; the report shows which arms are missing.
    """
    os.makedirs(run_dir, exist_ok=True)
    print("  " + " ".join(cmd) + "\n")
    # cwd=REPO_ROOT so the pipeline resolves its own relative paths as usual.
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    if result.returncode != 0:
        msg = f"Training failed with exit code {result.returncode} (run dir: {run_dir})."
        if abort_on_failure:
            raise SystemExit(msg)
        print(f"\n!!! ARM FAILED: {msg}\n!!! Continuing with the remaining arms.\n")
    return result.returncode


def load_history(run_dir, require_logvar=False):
    """Load a run's history, failing loudly if it predates the instrumentation."""
    path = os.path.join(run_dir, "training_history.json")
    if not os.path.exists(path):
        raise SystemExit(f"No history at {path}. Run without --analyze-only first.")
    with open(path, "r", encoding="utf-8") as fh:
        history = json.load(fh)
    if not history:
        raise SystemExit(f"{path} is empty.")

    # A history full of blanks would read like a negative result; refuse instead.
    needed = ["train_grad_norm"] + (["train_logvar", "val_logvar"] if require_logvar else [])
    missing = [k for k in needed if k not in history[-1]]
    if missing:
        raise SystemExit(
            f"{path} has no {', '.join(missing)} field(s). This history came from a "
            "pipeline build without the phase instrumentation -- re-run the training "
            "step against the current psi_full_pipeline.py."
        )
    return history


def mean_abs_step(series):
    """Mean absolute epoch-to-epoch change -- the volatility measure."""
    if len(series) < 2:
        return 0.0
    return sum(abs(series[i] - series[i - 1]) for i in range(1, len(series))) / (len(series) - 1)


def tail(history, window=TAIL_WINDOW):
    return history[-window:] if len(history) > window else history
