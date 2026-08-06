# Phase 2 report — single-judge v6e-1 shakedown (2026-08-06)

Orchestrated end-to-end by `tpu/pallas_arena/phase2/run_phase2.sbatch`
(sbatch job **3645906**, log `runs/pallas_arena/phase2-3645906.log`), which
enforces the hard rules: exactly one spot QR (fixed name
`sk7524-pallas-judge-shakedown`, never touches anything else), 2h
floundering cap, QR deleted on every exit path.

## Attempt 1 — bring-up GREEN, shakedown RED (RLIMIT bug), QR deleted

Measured bring-up numbers (decision inputs for the phase-3 fleet):

| Step | Result |
|---|---|
| Spot v6e-1 QR create → ACTIVE | **5.5 min** (10:24:29 → 10:30:02) despite the v6e-8 capacity crunch in-zone — v6e-1 spot capacity is fine |
| ssh reachable | ~30 s after ACTIVE |
| Provision (apt + uv venv py3.12 + `jax[tpu]==0.10.2` + TPU sanity `jax.devices()`) | **~50 s** (provision_judge: OK 10:31:21) |
| Artifact round-trip (scp code up, results/log down) | works |
| Always-delete trap | fired and removed the QR |

Shakedown outcome: **every TPU grading child died at jax init** with
`FAILED_PRECONDITION: TPU initialization failed: Couldn't mmap: Cannot
allocate memory` (`runs/pallas_arena/phase2-shakedown-results-3645906.json`).
Root cause: the grader's default `RLIMIT_AS` (16 GB — sized for CPU-side
Mosaic-compile blowups) is far below libtpu's virtual-address reservations,
so the child's TPU client could not mmap. The failure wrote the results
file in ~2 s, the wait loop saw it, artifacts were fetched, and the trap
deleted the QR — no idle spot resources were left behind.

Fix (committed **a8f110fc**, validated by code path, hot-pushed to the host
but ~2 min after the tmux launch had already started with the old tree):
`grader.grade` now resolves its rlimit default from `ARENA_RLIMIT_GB`, and
`phase2/shakedown.py` runs everything (direct grades AND the judge server's
children) with `ARENA_RLIMIT_GB=512`.

## Status

Phase 2 = **infrastructure green, shakedown numbers not yet captured**:
QR lifecycle, spot landing latency, provisioning, code shipping, artifact
fetch and cleanup are all proven; the RMSNorm invariant battery (noise
floor, ref-vs-ref 1.00±2%, regrade ±3%, peak-HBM, per-candidate cost,
POST /grade acceptance) still needs one clean run. Re-run is one command
(the fix is already in the tree the orchestrator ships):

    sbatch tpu/pallas_arena/phase2/run_phase2.sbatch

Held at one-QR-created pending an explicit go-ahead for a second QR (the
mandate capped this phase at one spot QR).
