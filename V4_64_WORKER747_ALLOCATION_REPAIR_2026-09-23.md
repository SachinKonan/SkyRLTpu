# v4-64 worker747 allocation repair, September 23, 2026

Worker747 was physically idle but unavailable to the managed-job queue because
job1596 retained its `current_cluster_name` reservation after its local attempt
failed. This was a scheduler bookkeeping problem, not evidence of a broken TPU.

## Evidence before repair

- Pool: `tpuswarm-v4-64-central2-qwen35-erdos`.
- Provider: worker747's TPU was READY and HEALTHY; SkyPilot's replica was READY.
- Managed job1596 (Qwen AC2) was PENDING/RECOVERING, bound to worker747, and
  repeatedly reported no idle replicas.
- Worker747's local job2, belonging to job1596, was `FAILED_DRIVER` with a recorded
  end time. All eight hosts had four TPU devices each, no device owners, no
  matching training/inference processes, and no active workload systemd units.
- The bounded recent API-request inventory found no pending or running execution
  request for worker747 before the repair.

The pool scheduler's fallback treats every nonterminal managed job's cluster
binding as a reservation. A recovering job can therefore block its own old,
otherwise idle replica. The inspected recovery implementation does not release
this binding before asking for another replica.

## Repair and scope

At 08:05:46 EDT, under the same pool file lock used by scheduling and an SQLite
write transaction, rechecked job1596's pool, worker, local job ID, and recovery
state. Cleared only its stale `current_cluster_name` using a conditional update.
Kept job1596 queued and retained its local job ID and history. No worker, job,
checkpoint, cache, or shared API process was stopped or deleted.

Within a second the scheduler assigned worker747 to existing Muse qubit job1600
and created execution request `ed02ad11-d1b7-4ab5-bd02-fb7bf66c8e8a`.
This establishes that the reservation was blocking allocation.

The execution request remained PENDING for over two minutes. After checking its
managed job ID, worker, run name, and immutable code hash against the existing
Muse1600 manifest, dispatched that same request through SkyPilot's locked
request executor. The request succeeded; no replacement managed job was created.

At 08:10 EDT, managed job1600 and worker-local job3 were RUNNING. The worker log
confirmed that all eight hosts had completed the routing library's release
build, so real startup work was executing. A resumed optimizer update has not
yet been verified. The existing continuation watcher still owns the transition
from this 15-step attempt to the authorized 25-step continuation.

This is an operational repair of the verified stale reservation. It does not
change the scheduler's general recovery logic; future stale bindings must not
be cleared merely because a controller reports PENDING or RECOVERING. Check
actual workload ownership and outstanding launch requests first.

## Local evidence

Files under `.science/routing-relaunch-20260921/`:

- `allocation-pool-now.txt`, `allocation-provider-now.json`
- `allocation-1596-controller.txt`
- `allocation-747-host0.json` through `allocation-747-host7.json`
- `allocation-747-repair.json`
- `allocation-747-post-repair.json`
- `allocation-747-dispatch.json`, when targeted execution dispatch was needed
- `allocation-747-workload-after.json`, `allocation-747-startup-after.json`
