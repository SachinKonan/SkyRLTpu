# Bounded Viz-Node SkyPilot Restart

Use the existing durable state on della-vis2, not a new SkyPilot HOME:

```bash
bash tpu/swarm/start_skypilot_bounded.sh --check
bash tpu/swarm/start_skypilot_bounded.sh --start
```

The launcher refuses an already-listening API. It starts only SkyPilot, not
the TPUSwarm application server. SkyPilot automatically recovers its existing
managed-job and pool controllers; pool targets and GCP reservations are not
manually changed. Do not use `sky check` to prepare a stopped server: that CLI
can implicitly start the default, unbounded local API.

Changing this launcher does not resize a running API. `--check` only validates
the proposed sizing; `--start` is for an already-stopped API, not a restart
command. Before stopping an existing API, establish how its in-flight requests
will be preserved: graceful shutdown can cancel provisioning requests after
its drain timeout. Durable state alone does not make a restart harmless.

## Applied Limits

- Deployment mode: 8 HTTP workers, 16 long-request workers, 14 short-request
  workers, and zero burst workers with this checkout's sizing rules.
- 8 managed-job controllers and capacity for all 6 existing pool controllers.
- `SKYPILOT_POD_CPU_CORE_LIMIT=8`, `SKYPILOT_POD_MEMORY_GB_LIMIT=25`, and
  `SKYPILOT_MEMORY_AWARE_WORKER_SIZING=true` govern internal sizing.
- CPU affinity is restricted to the first 4 CPUs already allowed to the
  launcher, inherited by the server and its children.
- BLAS/OpenMP thread counts are 1. The controller boot deadline remains 1800s.
- The in-flight replica limit remains 160; pool targets and scheduling are
  unchanged. No new admission gate or queue has been introduced.

These are worker-pool limits, not a hard total-process limit: HTTP supervisors,
queue processes, pool-controller children, and provisioning helpers also run.
The memory environment variable is a sizing budget, not an OS memory cap.
The eight-CPU sizing input deliberately permits more I/O-bound execution
processes; it does not grant eight physical CPUs. The four-CPU affinity remains
the CPU safeguard within this user's shared seven-core cgroup allowance.
The launcher validates calculated capacities before starting, failing closed
if a future source change alters these executor/HTTP worker counts, exceeds
eight managed-job controllers, or cannot fit six pools.

The initial 2-CPU/16-GiB sizing gave four long-request workers. All four became
occupied by indefinite TPU provisioning waits, leaving job 438's `sky.exec`
pending even though its TPU worker was available. The September 9 00:24Z restart
loaded 4 HTTP / 8 long / 8 short workers. The September 9 03:52Z restart applied
the increase to 8 HTTP / 16 long / 14 short, without enabling burst workers or
changing the scheduler. The short-worker count is configured capacity; workers
can start lazily rather than all appearing immediately in the process list.
It adds headroom, not a guarantee against starvation: provisioning and dispatch
still share the long-request queue. CPU affinity and numerical-library thread
limits remain unchanged. Recheck the live process and startup log after any
subsequent restart rather than assuming these settings are still loaded.

## Why These Limits Are Necessary

On 2026-09-08, this user's parent cgroup had `cpu.max = 700000 100000`, or
7 CPU cores shared by the user's processes. SkyPilot's cgroup detection in
`sky/utils/common_utils.py` checked the cgroup root instead of walking the
process's actual cgroup ancestors. It therefore missed the effective limit.
Default local mode sized thousands of possible short workers from host RAM,
allowed 1024 burst workers per request class, and recovered 64 job controllers.

Roughly 200 processes caused user-level CPU throttling. Apparent 100-300ms
filesystem latencies also included scheduling delays; node-wide idle CPU did
not rule out user-level throttling. Stopping the stack restored normal latency.
This launcher is an operational workaround; it does not fix cgroup detection.

Check the user cgroup's `cpu.stat` deltas, API `/api/health` and `/dashboard/`,
the six controller ports 20001-20006, and inherited CPU affinity after restart.
Do not equate responsive controllers with successful TPU allocation or training.

Logs are printed by the launcher under `tpuswarm-state/sky-api-bounded-*.log`.
The server is started in a detached session so it survives the launching shell.

## Preserving Existing Requests

Normal `sky.server.server` startup calls `reset_db_and_logs`: it deletes the
SQLite request database and transient client files even with a durable HOME.
The launcher alone therefore is not sufficient for an in-flight restart.

`skypilot_preserving_restart.py` is a one-shot, manifest-guarded startup
entrypoint selected by `SKYPILOT_RESTART_MANIFEST` in the bounded launcher. It
preserves the request database/client files, re-registers internal daemons, and
requeues the audited original launch and unexecuted exec request IDs. It refuses
unknown operations, changed records, foreign pools, live old API processes, and
replaying an exec that was already RUNNING. Seven focused tests cover its
request validation and transactional database update. This is an operator
recovery tool, not an unattended restart service or a generic exactly-once
execution guarantee.

Before replacing an API:

1. Hold `~/.sky/api_server/.creation.lock` using `filelock.FileLock` for the
   entire stop/start/health-check interval. Use the durable SkyPilot HOME.
   Another CLI can otherwise auto-start an unbounded API during the gap.
2. Inventory authenticated GCP queued resources and VMs, including names and
   creation times. Back up request, cluster, managed-job and pool databases.
3. Pause local pool/job controllers and quiesce supplemental executors. Record
   exact owned PIDs and process creation times, then take final SQLite backups
   and an active-request manifest. Fail if a workload exec is already RUNNING
   and its remote outcome is unknown. Do not signal TPU-host processes.
4. Replace only the audited API processes without invoking request-cancellation
   handlers. Start the preserving entrypoint with the bounded launcher. Keep
   the creation lock held until API health and original-ID requeue succeed.
5. Resume controllers and reconcile remote job IDs. Existing SDK connections
   can fail even when their server-side requests survive; a controller may
   submit a duplicate. Verify remote jobs.db before cancelling a never-started
   duplicate request or attaching a managed job to an existing remote job.
6. Verify every controller, API sizing, original GCP identities, and actual
   remote setup logs. Request SUCCEEDED means dispatch succeeded, not training.

All monitoring clients should export `SKYPILOT_DISABLE_LOCAL_API_SERVER=1`.
Do not modify or restart another agent's monitoring scripts without coordination.

## 2026-09-09 Restart Evidence

Audit directory under the durable HOME:
`.sky/recovery-audit/api-preserving-restart-20260909T001837Z/`.
It contains initial and quiesced SQLite backups, before/after provider JSON,
the process/request manifest, requeued IDs, and job-reattachment records. Treat
database backups as private: they can contain credentials.

- The first preserving start refused to run after a background `sky jobs queue`
  monitor auto-started an unbounded API and cleared the request DB. The quiesced
  backup was restored after stopping that accidental API and its newly spawned
  local controllers. No new launch/exec/down operations existed in that
  accidental API's request database.
- The subsequent start held the creation lock. Bounded API PID 3878086, log
  `tpuswarm-state/sky-api-bounded-20260909T002359Z.log`: 4 HTTP, 8 long,
  8 short, zero burst workers, affinity CPUs 0-3.
- All 234 original active launch/exec request IDs were preserved and requeued.
  All 98 owned GCP queued resources in the pre-restart inventory retained their
  names and creation times. All six pool controller ports returned HTTP 200.
- v5p workers 189/197 were already absent from the quiesced cluster DB; their
  dispatches failed with ClusterDoesNotExist. The pre-restart GCP snapshot
  already showed only worker 198 allocated in this v5p pool.
- Qwen managed job 438 dispatched as remote job 9 on v4 worker 95; gpt-oss job
  437 dispatched as remote job 2 on v5p worker 198. Both remote jobs.db records
  were RUNNING. Qwen logs reached XLA compilation; gpt-oss logs showed Ray
  startup. These observations do not establish successful generation/training.
- Old controller HTTP connections failed, so the managed records initially
  missed these successful dispatches. The unexecuted duplicate request for v4
  worker 97 was cancelled, the two confirmed remote job IDs were attached, and
  only the affected local controller processes were replaced without running
  their cleanup handlers. Their nonterminal jobs were put through the native
  `reset_job_for_recovery` scheduler path with task status preserved.
- Final controller verification: jobs 437 and 438 were claimed by surviving
  controller PID 946278; both logs report `Resuming task 0 from previous
  execution` and `Job status: JobStatus.RUNNING` with the same remote job IDs.
  Job 371 was also resumed on its unchanged worker 94 / remote job 4.
  Launch requests were progressing (118 PENDING, 100 WAITING, 8 RUNNING at
  the final check), rather than remaining in the old four-executor deadlock.

## 2026-09-09 Capacity Increase

Audit directory: `.sky/recovery-audit/api-capacity-restart-20260909T034649Z/`.

- API PID 2318739, startup log
  `tpuswarm-state/sky-api-bounded-20260909T035250Z.log`: 8 HTTP workers,
  16 long-request workers, configured capacity for 14 short-request workers,
  zero burst workers, affinity CPUs 0-3.
- Held the creation lock throughout the stop/start/health interval. Paused and
  resumed 29 local controller processes; replaced only 23 audited API-tree
  processes. No TPU-host processes were signalled by the restart operation.
- Preserved and requeued all 215 original active launch request IDs. There
  were no active workload exec requests at the quiesced snapshot. The other
  five active rows were internal daemons, which were freshly registered.
- API health and dashboard returned HTTP 200; all six controller
  `/autoscaler/info` endpoints returned HTTP 200. No controllers remained
  paused. Job 371 retained its RUNNING binding to v4 worker 94 / remote job 4;
  job 456 remained waiting for v5p capacity, without duplicate submission.
- All 23 owned central2-b and 29 owned east5-a queued resources retained their
  names and creation times, as did all 41 Asia queued resources. Europe had
  none. The three existing v4 TPU VMs remained in the provider inventory.
- Do not describe the full fleet as unchanged: three east5-b v6e requests
  disappeared between inventories, with three new requests in their place.
  GCP audit logs show the deletions completed at 03:47:28Z, 03:49:42Z, and
  03:50:54Z, before this API replacement. One was ACTIVE in the initial
  inventory. The audit includes the deletion records; the exact initiating
  cleanup paths were not fully traced during this restart.
- The added executor capacity does not fix controller health probes treating
  control-plane observation errors as replica failures. That remains a
  separate unresolved protection gap.

## 2026-10-04 Restart + requests.db VACUUM

Audit directory: `.sky/recovery-audit/api-vacuum-restart-20261004T204644Z/`
(driver `tpu/swarm/api_restart/api_vacuum_restart.py`: one subcommand per phase — prep, lock, stop, db, start, resume, verify).

- Taken with zero managed jobs and zero exec requests. Creation lock held by a
  detached holder from before the stop until after health + requeue.
- Pool controllers (5 groups, 17 procs) SIGSTOPped then SIGCONTed. The 8 idle
  jobs controllers were killed instead of paused, so the new API's startup
  recovery spawned fresh ones on the current tree (loads in-process pool
  dispatch, `SKYPILOT_POOL_INPROCESS_EXEC`, default on). API tree: 58 procs
  SIGKILLed.
- 14 stale non-launch rows (sky.jobs.queue_v2/logs/pool_status and one
  sky.down, 71–435 h old) blocked the preserving entrypoint's validation; they
  were marked CANCELLED after the stop (not via `sky api cancel`, which would
  signal the shared executor PIDs recorded on those rows).
- requests.db: logical backup via `VACUUM INTO` (1.64 GB, integrity ok), then
  in-place `PRAGMA auto_vacuum=INCREMENTAL; VACUUM`: 62.9 GB -> 1.64 GB,
  3,638 rows unchanged. state.db (0% free) and spot_jobs.db (5% free) were not
  worth compacting. Total API downtime 20:47:54Z–20:52:35Z.
- New API PID 3415535, log `tpuswarm-state/sky-api-bounded-20261004T205212Z.log`:
  8 HTTP / 16 long / 14 short (lazy) / 0 burst, affinity 0-3. All 36 audited
  launch request IDs preserved and requeued. All five controller
  `/autoscaler/info` endpoints 200. GCP: every owned queued resource kept its
  name and createTime (east5-a 37/37, central2-b 9/9, east5-b 16/16 plus one new
  request filed by the pool).

## 2026-10-05 Reattach after a stray auto-start

At 2026-10-04 21:40Z a client in the durable HOME hit the local-API auto-start
path (client unidentified; every client found afterwards had the disable env
var). The would-be server ran `reset_db_and_logs` before failing to bind:
`requests.db` was unlinked and an empty file created. The running API's 38
workers kept working on the deleted inode, so the API behaved normally but any
fresh reader (CLI-side checks, the nightly health report) saw an empty DB, and
a restart would have discarded all request state.

- Guard added to the fork: `check_local_api_server_enabled_or_raise` also
  refuses when `~/.sky/api_server/.disable-local-api-server` exists, and that
  file now exists in the durable HOME (test:
  `tests/unit_tests/test_disable_local_api_server_file.py`). It covers clients
  using this fork even without `SKYPILOT_DISABLE_LOCAL_API_SERVER=1`; clients on
  another SkyPilot install still need the env var.
- Recovery (audit `.sky/recovery-audit/api-reattach-restart-20261005T012522Z/`,
  driver `tpu/swarm/api_restart/api_reattach_restart.py`): pause pool and jobs controllers (live
  jobs existed), SIGSTOP the API tree, copy the deleted db+wal out via
  `/proc/<pid>/fd`, validate the copy (integrity ok, 3,497 rows, newest 18 s
  old, no running exec), then SIGKILL; install the copy, preserving start.
  Downtime 01:26:58Z–01:27:47Z. 61 launch request IDs requeued; 0 of 106 API
  DB handles on a deleted file afterwards; all owned queued resources kept
  name and createTime.
- How to spot this state: `ls -l /proc/<api-child>/fd | grep 'requests.db (deleted)'`.

## Restart Drivers (`tpu/swarm/api_restart/`)

Each driver runs one phase per invocation so every step can be checked
before the next; state is kept in the audit directory named by
`tpuswarm-state/api-restart/<driver>.current`. Run with the durable HOME and
`SKYPILOT_DISABLE_LOCAL_API_SERVER=1`, e.g.
`.venv/bin/python tpu/swarm/api_restart/api_plain_restart.py prep`, then
`lock`, `stop`, `db`, `start`, `resume`, `verify`.

- `api_plain_restart.py`: restart with live jobs (pauses pool and jobs
  controllers; aborts and thaws if a `sky.exec` is in flight). Use this to load
  new API-side code.
- `api_vacuum_restart.py`: restart with zero managed jobs plus an in-place
  `requests.db` VACUUM; kills idle jobs controllers so fresh ones load new code.
- `api_reattach_restart.py`: recover from a stray auto-start that unlinked
  `requests.db` while the API kept running on the deleted file.
- `swap_pool_controller.sh <pool>`: SIGKILL one pool controller's process group;
  the API's HA daemon respawns it on the current code within ~20 s.
- `reap_dead_qrs.sh <zone>`: one-off deletion of our own SUSPENDED/FAILED queued
  resources (the pool controllers' orphan sweep now does this every 10 min).
