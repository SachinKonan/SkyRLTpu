# Worker747 storage and stale reservation repair — September 24, 2026

Worker747's head disk had 10.3 GiB free. Qubit Gemma managed job1675's local
attempt13 failed the bootstrap requirement of 30 GiB free disk. All eight hosts
were idle, but job1675 retained its pool reservation while recovering, preventing
the scheduler from assigning this healthy slice to queued work.

## Completed repair

Under the pool scheduler's file lock, verified no pending/running Sky API request
for this worker, no live worker-local job, no TPU device owners, and no process
using the exact cleanup targets (command, executable, working directory, mapped
files, and open descriptors were checked with root visibility).

Removed only these categories from the head host:

- Retired Qwen/Muse qubit, AC2 Gemma, and RG-LRU environments/source copies, plus
  the disposable uv package cache.
- Old serving upload archives, after confirming each named adapter version has
  a nonempty sampler checkpoint in the same run's Central2 bucket.
- Interrupted `tinker-backup.partial` files, after the corresponding completed
  database passed SQLite `quick_check`.

Free head disk increased from 10.3 to **41.1 GiB**, reclaiming about **30.7 GiB**.
Training checkpoints, completed databases, future payloads, client/search state,
model/compilation caches on other filesystems, and circuit run artifacts were
preserved. No worker or TPU reservation was deleted. No new storage transfers
were required for this repair.

At 10:52 EDT, a fresh audit of all eight hosts found no TPU device owners or
recognized training/inference/install/transfer processes. Every host had at least
41.0 GiB free. Head-local job13 remained FAILED with a recorded end time.

Under the same pool lock and an SQLite write transaction, conditionally cleared
only job1675's stale `current_cluster_name`, after matching its pool, worker747,
local job13, and recovery state. Job1675 remains queued with its history intact.
Job priorities and other job bindings were unchanged.

## Allocation result

The scheduler assigned worker747 to existing AC2 Gemma job1686 immediately after
the reservation was cleared. Its `sky.exec` request
`b7dfef0c-49aa-40e1-ae5f-55295b82c3eb` remained PENDING for more than two minutes.
After checking the exact worker, managed job ID, run name, and immutable bundle
hash against the regionalized job1686 manifest, invoked SkyPilot's locked native
executor for that same request. It SUCCEEDED; no new managed job was submitted.

At 10:55 EDT, worker-local job14 was RUNNING for AC2 Gemma1686, with 40.9 GiB free
on the head disk. Qubit Gemma1675 and Qwen1664 remain queued for capacity; the
existing AC2 priorities were preserved. Training updates have not yet been
verified for this new attempt.

This repairs the verified worker blockage; it does not change general cache
retention or scheduler recovery behavior. Allocation and actual workload startup
must be verified separately; no resumed optimizer step is claimed here.

## Evidence and guarded repair scripts

`.science/routing-relaunch-20260921/gemma-central2-cache-fix-20260924/cleanup747/`
contains the before/after host audits, `durable-checkpoints.json`,
`disk-cleanup-receipt.json`, `retired-exports-receipt.json`, and
`reservation-repair.json`. The scripts assert worker ownership and idle state;
they are specific to this failed attempt and are not general-purpose cleanup.
