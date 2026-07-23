# Phase checks — master summary

Generated 2026-07-21T13:43:32 · total runtime 0.0 min

| phase | question | verdict | exit | runtime |
|---|---|---|---:|---:|
| 1 | Log-variance observation - is the uncertainty head saturating? | H1 REJECTED | ok | 0.0 min |
| 2 | Grad-clip sweep - is unconditional clipping capping learning? | H2 REJECTED - clip is not the lever | ok | 0.0 min |
| 3 | Learning-rate sweep - is 'bigger LR performs worse' real? | H3 INCONCLUSIVE - budget-limited | ok | 0.0 min |

## Peak resource use

| resource | peak observed |
|---|---:|
| host RAM used | 797 MB |
| GPU memory used | 66 MB |

Sampled every 3s across the whole sequence. Peaks are the reason this
exists: Phase 2 lost an arm to an unexplained CUDA fault on a 12.2 GB host.

## Full log

`logs/run_all_20260721_134328.log` — every subprocess line, timestamped to
the millisecond, plus periodic memory samples.

## Per-phase reports

- [PHASE1_REPORT.md](PHASE1_REPORT.md) — Log-variance observation - is the uncertainty head saturating?
- [PHASE2_REPORT.md](PHASE2_REPORT.md) — Grad-clip sweep - is unconditional clipping capping learning?
- [PHASE3_REPORT.md](PHASE3_REPORT.md) — Learning-rate sweep - is 'bigger LR performs worse' real?
