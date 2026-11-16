# Inkling-2-Small Erdős runs on Tinker (supervised, della-vis1)

Client: this repo (`tpu/run_ttd_ensemble.py`) + the `discover` client library, on the real Tinker service.
Grading: Harish's 6 v5e gradefarm hosts (v3.1: 1 CPU / 1 GiB, 1000 s, seeded), reached through SSH forwards via the
OD v4 host 0. Everything lives under `/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/` (`venv/`, `code/`, `ops/`); run
directories under `/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs/`. Design and settings: `../../../tinker-inkling-runs/REVIEW.md`
(outside the repo).

| File | Role |
|---|---|
| `deploy.sh SKYRL_COMMIT DISCOVER_COMMIT` | pinned code snapshot + ops copy (+ manifest `ops/DEPLOYED`) |
| `launch_run.sh NAME STEPS SEED` | one run with the experiment settings; forensics in `<run>/diag/` |
| `run_service.sh run/stop/tunnel/watch/status` | systemd --user units: `tinker-run-NAME` (restart on failure, resumes), `tinker-tunnel`, `tinker-watch` |
| `tinker_watch.py` | hang watchdog (no progress 120 min -> restart), email on crash/restart/finish/low grading capacity, registry refresh |
| `tunnel_loop.sh` | SSH forwards 127.0.0.1:27861-6 -> gradefarm-harish-1..6, reconnecting |
| `diag_run.py` | runs the client with fault/audit/heartbeat logging |
| `check_farms.py` | gradefarm readiness / capacity through the forwards |

After a vis1 reboot: `run_service.sh tunnel`, `run_service.sh watch WATCH_CHECK_S=300 WATCH_HANG_MIN=120`, then
`run_service.sh run NAME STEPS SEED [VAR=VALUE ...]` for each unfinished run (each resumes from its checkpoint and tree).
Failure drill (2026-11-16): kill -9 mid-run, SIGSTOP hang, tunnel kill — all recovered and resumed at the right step.
