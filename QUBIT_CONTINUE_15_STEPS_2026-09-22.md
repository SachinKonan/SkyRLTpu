# Qubit PWC continuation to 15 total steps

The user authorized five additional steps for all three models on September 22. Qwen and Muse had finished ten steps; Gemma was still completing its original ten-step run. Each continuation targets 15 total optimizer steps and requires a saved checkpoint/search snapshot at step 10 or later.

| Model | Original job | Continuation | Placement |
|---|---:|---|---|
| Muse | 1482, succeeded at 10 | 1589 failed; paused, not resubmitted | Initially 727, last failed attempt on 744 |
| Qwen | 1483, succeeded at 10 | 1590 | v4-64 worker 734 |
| Gemma | 1493, still running | Prepared; submit automatically after 1493 succeeds at step 10 | Existing v4-64 pool |

Pool: `tpuswarm-v4-64-central2-qwen35-erdos`. Managed job names remain equal to the original runtime run IDs so discovery recognizes them. Durable run IDs, GCS paths, optimizer checkpoints, and search pools remain unchanged.

## What changed

- `NUM_EPOCHS`: 10 to 15.
- `resume_min_checkpoint_step`: 0 to 10; strict checkpoint resume stays enabled.
- Local root gains `-continue15`, and the sick-marker path follows it. A fresh local namespace prevents a stale snapshot left on a reused worker from overriding the newer durable run state. The executor retains its existing cache reuse/admission mechanism.

Each immutable deployment bundle is copied from the original job's SHA-pinned bundle, with only its selected profile JSON replaced. Every other archived file was checked byte-for-byte unchanged. No model, loss, optimizer, sampling, grader, inference-engine, or farm-scheduler changes are included. These launches therefore do not deploy the separately diagnosed farm-performance fixes.

Validation ran on Slurm CPU job 14283509 and passed for all three profiles and bundles. Before submitting Qwen/Muse, the launcher verified source-job success, step-10 metrics, a matching nonempty search snapshot, the checkpoint index, durable training archives including the index's final checkpoint, and the database backup. It also checked service-account identity, healthy allocated v4-64 nodes, and the active pool inventory.

## Deferred Gemma launch

Watcher: `.science/routing-relaunch-20260921/submit_extend15.py --watch-gemma`.

Process receipt: `.science/routing-relaunch-20260921/extend15/gemma-watch-process.json` (initial PID 16799 on della-vis2). Status and log are `gemma-watch-status.json` and `gemma-watch.log` in that directory. The watcher runs detached, checks every 60 seconds, and expires after 72 hours. It waits through running/recovering states, requires successful completion and durable step-10 state, and refuses to launch if another active job owns the same run ID. Terminal source failures and ambiguous submissions stop it for inspection. This is a local background process, not a provisioned SkyPilot dependency job.

The watcher is protected by a single-instance lock. Each submission has a separate lock, pre-launch intent, and receipt; an ambiguous launch is never retried automatically. Gemma's current computation is left running.

## Submission evidence

Private evidence: `.science/routing-relaunch-20260921/extend15/`.

- `prepared.json`: all three bundle hashes, task paths, and exact configuration changes.
- `1482/submitted.json` and `1483/submitted.json`: Muse 1589 / Qwen 1590 receipts.
- Each model directory contains its original/new archive, manifest, task YAML, and pre-launch checkpoint/inventory evidence.
- `started-status.json`: most recent startup observation.

SkyPilot initially left the two existing `sky.exec` requests pending. They were dispatched through SkyPilot's normal locked execution wrapper after verifying exact job/cluster/request identities; both requests completed successfully. No duplicate jobs were submitted and the API server was not restarted. Request IDs: Muse `4d124be2-8325-465e-a462-285a52baf43b`; Qwen `0e087192-63a0-4da6-9af8-9ddbd516f942`.

## Latest verified progress and scope

The subsequent September 22 evening checks confirmed Qwen 1590 completed step 11, with best reward 0.5485872758724842. Gemma 1493 completed step 7, with best reward 0.5515663003177601; its step-10-to-15 watcher remained armed. The current focus is continuing Qwen and Gemma. These are recorded observations, not live status guarantees.

Muse continuation 1589 failed without a new completed update. Its initial attempt failed the host RAM-cache admission check; the proposed reduction from 128 to 96 GiB was not applied. Muse is now paused at its preserved step-10 checkpoint, best reward 0.5253055890789864. Its continuation profile is retained as the historical submitted configuration, not a validated retry recipe.

Latest comparison: [best saved programs and cached baselines](tpu/science/results/qubit-best-vs-cached-20260922-latest/README.md). That directory records Qwen step 11, Gemma step 7, and Muse step 10, with source hashes, saved evaluation observations, and all 72 case counts. The earlier `qubit-best-vs-cached-20260922` directory is a separate earlier snapshot. No candidate or baseline was rerun for these comparisons.

## Farm recovery observation

Qwen was verified using farm 1484 on v4-32 worker 183 (`http://10.130.0.208:24800`), with matching adapter hashes, four ready engines, and active remote generation.

Gemma lost its previous farm at `10.130.0.120:24800`. Discovery supplied replacement addresses, but the current sampling phase continued locally. Inspection of the deployed bundle established why: an unacknowledged release sets `uncertain_until` to the current time plus the 300-second lease duration and 10-second RPC timeout. `RunBorrower._prepare()` calls `reserve()` once per sampling phase; acquisition skipped during that interval is not retried within the phase. A later sampling phase can retry. No borrowing code or live leases were modified during this investigation.

## Spare-worker cleanup

At the cleanup snapshot, v4-64 workers 727 and 748 were READY and unassigned. All eight hosts on each were audited. No leftover trainer or inference processes were found; SkyPilot's Ray processes were preserved. Owned, unused executor tmpfs caches were normally unmounted under admission/run locks after process and mount-user checks, reclaiming 121.75 GiB across worker 727 and 366.78 GiB across worker 748. Failed runtime-service records were reset, and head-host package caches were cleaned with `uv cache clean`.

Post-cleanup checks found no executor RAM mounts or leftover workload processes on those 16 hosts. Head disk free space was 30.65 GiB on 727 and 60.85 GiB on 748; 727 remained close to the 30-GiB admission threshold. Checkpoints, logs, installed environments, TPU capacity, and active workloads were preserved. These worker assignments are historical and must be refreshed before any further cleanup.

Private audit receipts: `.science/routing-relaunch-20260921/spare-cleanup-result.json`, `spare-cleanup/`, and `spare-cleanup-after/`.
