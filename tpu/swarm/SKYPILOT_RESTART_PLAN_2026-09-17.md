# Proposed API-server capacity restart, 2026-09-17 (NOT executed; awaiting operator go)

Approved in principle by the operator on 2026-09-17: (1) raise long-request workers 16 -> 32,
(2) stop zone-cap quota failures from re-running a full provision attempt every 30 s,
(4) lengthen the paused-launch resume interval. (3) VACUUM of requests.db declined ("it's fine").
Hard constraint: no running TPU instance, managed job, or GCP queued resource (reservation) may be
disturbed. Restart only with explicit go.

## What a restart touches, and what it does not

Verified on the live host (della-vis2, user cgroup `user-374192.slice`):

- API server tree: launcher PID 2318739 (`tpu/swarm/skypilot_preserving_restart.py --deploy`),
  56 processes, 13.1 GB RSS, CPU affinity 0-3, env `SKYPILOT_POD_CPU_CORE_LIMIT=8`,
  `SKYPILOT_POD_MEMORY_GB_LIMIT=25`, `SKYPILOT_MEMORY_AWARE_WORKER_SIZING=true`.
- Pool controllers (5 x `sky.serve.service`) and managed-job controllers (8 x `sky.jobs.controller`)
  are **separate process groups with ppid 1**. Stopping the API tree does not stop them. They are
  paused (SIGSTOP) for the swap and resumed after, per the documented procedure.
- GCP queued resources and TPU VMs live on GCP; the server holds no lock on them. The 2026-09-09
  capacity restart (audit dir `.sky/recovery-audit/api-capacity-restart-20260909T034649Z/`,
  `provider-comparison.json`) preserved all 29 east5-a and 23 central2-b requests by name and
  creation time. The one caveat: 3 east5-b requests that were mid-PROVISIONING were deleted and
  re-filed by cleanup paths that fired *before* the swap. Today east5-b has 15 PROVISIONING
  requests; those are the exposure. WAITING_FOR_RESOURCES and ACTIVE requests were untouched.
- In-flight `sky.launch` requests are preserved, not cancelled: `skypilot_preserving_restart.py`
  resets them to PENDING and requeues the original request IDs on start (manifest-guarded; refuses
  if any old API process is alive, if records changed, or if a workload `sky.exec` was RUNNING).
- User cgroup: `cpu.max` 7 cores, `memory.max` 322 GB, `memory.current` 32 GB. Memory is not a
  real constraint; the 25 GB figure is only a sizing input.

## Change 1: 32 long-request workers

Sizing is `min(cpu_limit * _CPU_MULTIPLIER_FOR_LONG_WORKERS, memory-based)` in
`sky/server/config.py` (`_CPU_MULTIPLIER_FOR_LONG_WORKERS = 2`, `LONG_WORKER_MEM_GB = 0.4`,
`_MAX_MEM_PERCENT_FOR_BLOCKING = 0.6`, consolidation-mode headroom 4 GB). HTTP workers =
cpu_limit. Job-controller count and pool slots also scale with the memory budget
(`controller_utils._get_parallelism`). Dry runs (`jobs/f6d76b15/tmp/size_check.py`,
`size_check2.py`, read-only):

| cpu limit | mem budget | multiplier | http | long | short | job ctrls | pool slots |
|---|---|---|---|---|---|---|---|
| 8 (today) | 25 | 2 | 8 | 16 | 14 | 8 | 6 |
| 16 | 25 | 2 | 16 | 11 | 11 | 8 | 6 |
| 16 | 40 | 2 | 16 | 27 | 25 | 13 | 10 |
| 8 | 30 | 4 | 8 | 21 | 19 | 10 | 8 |
| 8 | 36 | 4 | 8 | 27 | 25 | 12 | 10 |

No knob reaches exactly 32 long workers without also doubling HTTP workers or inflating
controller counts. **Recommended:** add an explicit override in the fork,
`SKYPILOT_LONG_WORKERS` (read in `compute_server_config`, replaces `max_parallel_for_long` when
set), keep cpu limit 8 / mem 25 / 8 HTTP / 14 short / 8 job controllers, and set it to 32 in
`start_skypilot_bounded.sh`. Update the launcher's assertion `garanteed_parallelism == 16` to 32.
Extra resident memory: 16 x 0.4 GB = 6.4 GB nominal (tree today 13.1 GB). Extra CPU: none at
rest; long workers are I/O-bound (gcloud waits), affinity stays 0-3.

## Change 2: zone-cap failures stop churning workers

Path: `sky/backends/cloud_vm_ray_backend.py`. `QuotaFailure ... in zone` blocks the zone (l.~717),
the single-zone pool launch then has no launchable zone -> `ResourcesUnavailableError` ->
`retry_until_up` raises `ExecutionRetryableError(retry_wait_seconds=_RETRY_UNTIL_UP_INIT_GAP_SECONDS=30)`
(l.3573). The executor (`sky/server/requests/executor.py` l.434-475) sleeps that 30 s in the
**monitor thread**, so the slot is released during the wait; the cost is that every 30 s each
zone-capped launch re-enters the queue and burns a long worker for a full GCP provision attempt
(tens of seconds of gcloud calls that end in the same quota error). With ~13 such launches on v5p
this is a large share of worker time. Correction to the 09-16 note: the slot is not held for the
whole 30 s, it is held for each attempt.

**Patch:** in the `QuotaFailure` `in zone` branch, when the quota name is
`QueuedResourcePerProjectPerZone` and the launch is in a request context, raise
`ExecutionPausedError(..., retry_wait_seconds=300)` (same class used for infinite
queued-resource waits, `sky/exceptions.py:737`) instead of blocking the zone, so the request parks
for 5 min and one attempt per 5 min replaces one per 30 s. No behaviour change for other quota
errors or for non-pool launches.

## Change 4: paused-launch resume interval 60 s -> 300 s

`sky/provision/gcp/instance_utils.py:1720` (`wait_for_queued_resource.pause_if_scheduled`):
`ExecutionPausedError(..., retry_wait_seconds=60)`. Each resume re-runs the launch up to the
queued-resource `get`, then pauses again. ~100 parked launches x 1/min = a steady worker load for
no information (GCP grants land in waves hours apart). Change 60 -> 300. Effect on grant latency:
a granted slice is noticed within 5 min instead of 1 min; setup takes 5-8 min anyway.

## Procedure (from tpu/swarm/SKYPILOT_BOUNDED_RESTART.md, unchanged)

1. Apply the three fork patches + launcher edits in the main checkout's submodule
   (`third_party/TPUSwarm/third_party/skypilot`); run the fork's unit tests for config/executor.
2. `bash tpu/swarm/start_skypilot_bounded.sh --check` must print long=32 with everything else equal.
3. Hold `~/.sky/api_server/.creation.lock`; inventory GCP queued resources + VMs per zone
   (before JSON); back up `requests.db`, `state.db`, `spot_jobs.db`, `serve/services.db`.
4. SIGSTOP the 5 pool + 8 job controllers (record PIDs + create times); final quiesced backups;
   write the manifest (active requests, api_processes, controllers, home).
5. Stop only the audited API-tree processes without running cancellation handlers (SIGKILL the
   tree, as on 09-09); verify none alive.
6. `SKYPILOT_RESTART_MANIFEST=<dir>/manifest.json bash tpu/swarm/start_skypilot_bounded.sh --start`;
   wait for `/api/health` 200 and `requests-requeued.json`.
7. SIGCONT controllers; check the 5 controller `/autoscaler/info` endpoints; re-inventory GCP and
   diff by name + creation time (expect zero missing in east5-a/central2-b; watch the 15 east5-b
   PROVISIONING requests); confirm the 16 MoM managed jobs keep their state and no duplicate
   launches were filed.

Expected downtime for CLI/SDK calls: 1-3 min. Expected effect on running TPU workers and their
training processes: none (never signalled). Expected effect on pending reservations: none for
WAITING/ACTIVE; PROVISIONING ones may be re-filed as on 09-09.

Rollback: revert the launcher assertion and the env override; the three code patches are
behaviour-preserving when the env var is unset except for the two interval changes, which can be
reverted independently.
