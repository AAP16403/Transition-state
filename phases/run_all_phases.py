#!/usr/bin/env python3
"""Run every phase check in sequence, logging everything.

Executes phase_check_1 -> phase_check_2 -> phase_check_3 back to back, teeing all
output to a timestamped master log with per-line timestamps, and sampling host RAM
and GPU memory throughout.

Why the resource sampling: Phase 2 lost an arm to `CUDA error: unknown error` at
torch.cuda.synchronize() with no diagnostic trail. The host has 12.2 GB RAM and the
GPU 6.1 GB VRAM, and a run was observed at 10.9 GB RSS shortly before that crash.
Sampling both means the next such failure has a memory trace attached instead of
being unexplainable.

A failing phase does not stop the sequence -- each phase is independent and a
driver-level fault in one says nothing about the others.

Usage
-----
    python run_all_phases.py                 # resume: skips completed runs
    python run_all_phases.py --analyze-only  # rebuild every report, no training
    python run_all_phases.py --force         # re-run everything from scratch
    python run_all_phases.py --only 2 3      # just these phases

Writes phases/logs/run_all_<timestamp>.log  and  phases/MASTER_REPORT.md
"""

import argparse
import datetime as dt
import os
import re
import subprocess
import sys
import threading
import time

import phase_common as pc

LOG_DIR = os.path.join(pc.PHASES_DIR, "logs")
SAMPLE_SECONDS = 30

PHASES = [
    (1, "phase_check_1.py", "PHASE1_REPORT.md",
     "Log-variance observation - is the uncertainty head saturating?"),
    (2, "phase_check_2.py", "PHASE2_REPORT.md",
     "Grad-clip sweep - is unconditional clipping capping learning?"),
    (3, "phase_check_3.py", "PHASE3_REPORT.md",
     "Learning-rate sweep - is 'bigger LR performs worse' real?"),
]


class Log:
    """Timestamped tee to file and console, safe across the sampler thread."""

    def __init__(self, path):
        self.path = path
        self._lock = threading.Lock()
        self._fh = open(path, "a", encoding="utf-8", buffering=1)

    def write(self, tag, message):
        stamp = dt.datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{stamp}] [{tag:<7}] {message}"
        with self._lock:
            self._fh.write(line + "\n")
            # errors=replace: the pipeline emits Angstrom signs and box-drawing that
            # a cp1252 console cannot encode; the log file itself stays UTF-8.
            sys.stdout.write(line.encode(sys.stdout.encoding or "utf-8",
                                         errors="replace").decode(sys.stdout.encoding
                                                                  or "utf-8") + "\n")
            sys.stdout.flush()

    def rule(self, title):
        self.write("=" * 7, "")
        self.write("PHASE", f"### {title} ###")

    def close(self):
        with self._lock:
            self._fh.close()


def read_ram():
    """Host RAM in MB from /proc/meminfo (total, available). WSL always has it."""
    try:
        with open("/proc/meminfo") as fh:
            info = {}
            for line in fh:
                k, _, v = line.partition(":")
                info[k] = int(v.split()[0]) // 1024
        return info.get("MemTotal"), info.get("MemAvailable")
    except OSError:
        return None, None


def read_gpu():
    """GPU (used_MB, total_MB) via nvidia-smi, or (None, None) if unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if out.returncode != 0:
            return None, None
        used, total = out.stdout.strip().splitlines()[0].split(",")
        return int(used), int(total)
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None, None


class Sampler(threading.Thread):
    """Background RAM/VRAM sampler. Records peaks so a crash leaves a memory trace."""

    def __init__(self, log, interval):
        super().__init__(daemon=True)
        self.log, self.interval = log, interval
        self._stop = threading.Event()
        self.peak_ram_used = 0
        self.peak_gpu_used = 0
        self.samples = 0

    def run(self):
        # Sample immediately, then on the interval. Waiting first would leave any run
        # shorter than one interval with zero samples, and a reported peak of 0 MB
        # reads as "used no memory" rather than "never measured".
        self._sample()
        while not self._stop.wait(self.interval):
            self._sample()

    def _sample(self):
        total, avail = read_ram()
        gpu_used, gpu_total = read_gpu()
        parts = []
        if total and avail:
            used = total - avail
            self.peak_ram_used = max(self.peak_ram_used, used)
            parts.append(f"RAM {used:,}/{total:,} MB ({100 * used / total:.0f}%)")
        if gpu_used is not None:
            self.peak_gpu_used = max(self.peak_gpu_used, gpu_used)
            parts.append(f"VRAM {gpu_used:,}/{gpu_total:,} MB "
                         f"({100 * gpu_used / gpu_total:.0f}%)")
        if parts:
            self.samples += 1
            self.log.write("RES", " | ".join(parts))

    def peak_text(self, attr):
        """Never report a bare 0: distinguish 'measured as 0' from 'never measured'."""
        return f"{getattr(self, attr):,} MB" if self.samples else "not sampled"

    def stop(self):
        self._stop.set()


def log_environment(log, sample_seconds):
    log.rule("ENVIRONMENT")
    log.write("ENV", f"timestamp      {dt.datetime.now().isoformat(timespec='seconds')}")
    log.write("ENV", f"python         {sys.version.split()[0]} ({sys.executable})")
    log.write("ENV", f"cwd            {os.getcwd()}")
    log.write("ENV", f"repo root      {pc.REPO_ROOT}")
    log.write("ENV", f"pipeline       {pc.PIPELINE}")
    log.write("ENV", f"sample cache   {pc.DEFAULT_SAMPLE_CACHE} "
                     f"(exists={os.path.exists(pc.DEFAULT_SAMPLE_CACHE)})")
    try:
        import torch
        log.write("ENV", f"torch          {torch.__version__} cuda={torch.version.cuda} "
                         f"available={torch.cuda.is_available()}")
        if torch.cuda.is_available():
            log.write("ENV", f"gpu            {torch.cuda.get_device_name(0)}")
    except ImportError as e:
        log.write("ENV", f"torch          NOT IMPORTABLE ({e})")
    total, avail = read_ram()
    if total:
        log.write("ENV", f"host RAM       {avail:,} MB available of {total:,} MB")
    gpu_used, gpu_total = read_gpu()
    if gpu_used is not None:
        log.write("ENV", f"GPU memory     {gpu_used:,} MB used of {gpu_total:,} MB")
    try:
        rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=pc.REPO_ROOT,
                             capture_output=True, text=True, timeout=10)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=pc.REPO_ROOT,
                               capture_output=True, text=True, timeout=10)
        n_dirty = len([x for x in dirty.stdout.splitlines() if x.strip()])
        log.write("ENV", f"git            {rev.stdout.strip()} ({n_dirty} modified file(s))")
    except (OSError, subprocess.TimeoutExpired):
        log.write("ENV", "git            unavailable")
    # The actual interval in use, not the module default -- a log that misreports its
    # own sampling rate makes every memory trace in it untrustworthy.
    log.write("ENV", f"sampling every {sample_seconds}s")


def run_phase(log, number, script, extra):
    """Run one phase, streaming every output line into the log. Returns (rc, secs)."""
    cmd = [sys.executable, "-u", os.path.join(pc.PHASES_DIR, script)] + extra
    log.write("CMD", " ".join(cmd))
    started = time.time()
    proc = subprocess.Popen(
        cmd, cwd=pc.PHASES_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
    )
    for line in proc.stdout:
        log.write(f"P{number}", line.rstrip())
    proc.wait()
    elapsed = time.time() - started
    tag = "OK" if proc.returncode == 0 else "FAILED"
    log.write("RESULT", f"phase {number} {tag}: exit={proc.returncode} "
                        f"elapsed={elapsed / 60:.1f} min")
    return proc.returncode, elapsed


def read_verdict(report_name):
    """Pull the '**Verdict: ...**' line out of a phase report."""
    path = os.path.join(pc.PHASES_DIR, report_name)
    if not os.path.exists(path):
        return "no report produced"
    with open(path, encoding="utf-8") as fh:
        m = re.search(r"\*\*Verdict:\s*(.+?)\*\*", fh.read())
    return m.group(1).strip() if m else "verdict not found in report"


def write_master_report(results, log_path, sampler, sample_seconds):
    path = os.path.join(pc.PHASES_DIR, "MASTER_REPORT.md")
    rows = ["| phase | question | verdict | exit | runtime |", "|---|---|---|---:|---:|"]
    for number, _, report, question, rc, secs in results:
        status = "ok" if rc == 0 else f"**{rc}**"
        rows.append(f"| {number} | {question} | {read_verdict(report)} | {status} | "
                    f"{secs / 60:.1f} min |")
    total = sum(r[5] for r in results) / 60
    body = f"""# Phase checks — master summary

Generated {dt.datetime.now().isoformat(timespec='seconds')} · total runtime {total:.1f} min

{chr(10).join(rows)}

## Peak resource use

| resource | peak observed |
|---|---:|
| host RAM used | {sampler.peak_text("peak_ram_used")} |
| GPU memory used | {sampler.peak_text("peak_gpu_used")} |

Sampled every {sample_seconds}s across the whole sequence. Peaks are the reason this
exists: Phase 2 lost an arm to an unexplained CUDA fault on a 12.2 GB host.

## Full log

`{os.path.relpath(log_path, pc.PHASES_DIR)}` — every subprocess line, timestamped to
the millisecond, plus periodic memory samples.

## Per-phase reports

""" + "\n".join(f"- [{r[2]}]({r[2]}) — {r[3]}" for r in results) + "\n"
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(body)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default=pc.DEFAULT_DATA_DIR)
    parser.add_argument("--sample-cache", default=pc.DEFAULT_SAMPLE_CACHE)
    parser.add_argument("--analyze-only", action="store_true",
                        help="Rebuild every report without training")
    parser.add_argument("--force", action="store_true",
                        help="Re-run every arm even if already complete")
    parser.add_argument("--only", nargs="+", type=int, metavar="N",
                        help="Run only these phase numbers")
    parser.add_argument("--sample-seconds", type=int, default=SAMPLE_SECONDS)
    args = parser.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(LOG_DIR, f"run_all_{stamp}.log")
    log = Log(log_path)

    selected = [p for p in PHASES if not args.only or p[0] in args.only]
    if not selected:
        raise SystemExit(f"--only {args.only} matched no phases (valid: 1, 2, 3).")

    extra = ["--data-dir", args.data_dir, "--sample-cache", args.sample_cache]
    if args.analyze_only:
        extra.append("--analyze-only")
    if args.force:
        extra.append("--force")

    log.rule("PHASE CHECK SEQUENCE")
    log.write("PLAN", f"log file       {log_path}")
    log.write("PLAN", f"phases         {', '.join(str(p[0]) for p in selected)}")
    log.write("PLAN", f"mode           "
                      f"{'analyze-only' if args.analyze_only else ('force' if args.force else 'resume')}")
    log_environment(log, args.sample_seconds)

    sampler = Sampler(log, args.sample_seconds)
    sampler.start()

    results = []
    sequence_started = time.time()
    try:
        for number, script, report, question in selected:
            log.rule(f"PHASE {number} — {question}")
            rc, secs = run_phase(log, number, script, extra)
            verdict = read_verdict(report)
            log.write("VERDICT", f"phase {number}: {verdict}")
            if rc != 0:
                log.write("ERROR", f"phase {number} exited {rc} — continuing with the "
                                   "remaining phases (a fault in one says nothing "
                                   "about the others)")
            results.append((number, script, report, question, rc, secs))
    finally:
        sampler.stop()
        sampler.join(timeout=5)

    log.rule("SUMMARY")
    for number, _, report, question, rc, secs in results:
        log.write("SUMMARY", f"phase {number}: exit={rc} {secs / 60:5.1f} min "
                             f"| {read_verdict(report)}")
    log.write("SUMMARY", f"peak host RAM used  {sampler.peak_text('peak_ram_used')} ({sampler.samples} samples)")
    log.write("SUMMARY", f"peak GPU memory     {sampler.peak_text('peak_gpu_used')}")
    log.write("SUMMARY", f"total elapsed       {(time.time() - sequence_started) / 60:.1f} min")

    master = write_master_report(results, log_path, sampler, args.sample_seconds)
    log.write("SUMMARY", f"master report -> {master}")
    log.write("SUMMARY", f"full log      -> {log_path}")
    log.close()

    # Non-zero if any phase failed, so the launcher's ERRORLEVEL is meaningful.
    sys.exit(1 if any(r[4] != 0 for r in results) else 0)


if __name__ == "__main__":
    main()
