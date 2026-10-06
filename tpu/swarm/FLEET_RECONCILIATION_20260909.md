# Fleet Reconciliation, September 9

Live controller state is separate from GCP queued resources and running jobs.
Do not infer an allocated TPU from a PROVISIONING record or a pool target.

## Targets

| Pool | Desired workers | GCP requests at verification |
| --- | ---: | ---: |
| v4-32 central2-b | 5 | 5 |
| v4-64 central2-b | 29 | 18 |
| v5p-32 east5-a | 48 | 29 |
| v6e-32 Asia northeast1-b | 43 | 40 |
| v6e-32 east5-b | 32 | 9 |
| v6e-32 Europe west4-a | 2 | 0 |

The user authorized maximizing the pools while preserving reservations. Five
v4-32 reservations already existed, so all five were retained. With those
160 cores retained, 29 v4-64 slices are the whole-slice ceiling within the
shared 2,048-core quota. The v5p ceiling is 48 slices within 1,536 cores.
These ceilings do not subtract other users' future allocations. GCP enforces
the shared quota; neither target guarantees an allocation.

At inventory, central2-b had all 100 queued-resource slots occupied, east5-b
had all 40 occupied, and east5-a's requests accounted for its 1,536-core
quota. Asia's 48 request slots, minus five other requests, allowed 43 owned
requests. Europe had 56 of 64 v6e Spot units requested, insufficient for
another v6e-32. Other users' failed/suspended requests were not deleted.
The generic single-pool YAML remains a one-worker template; the regional
Asia target is persisted in the durable service version, not that template.

## Changes Applied

- Removed ghost v5p replicas 189/198 after checking provider absence, no
  handles, and no active operations. The native controller purge path avoided
  any cloud deletion for these records.
- Recovered 21 existing WAITING_FOR_RESOURCES reservations without changing
  their cloud identities: v5p 200, 209, 212, 213, 216, 220, 236, 242, 243, 246;
  v4-64 44 and 65-73; v4-32 98. These now have active provisioning management,
  not fabricated READY status.
- Retired ten unallocated v5p placeholders to keep its target at 48. Their
  25 redundant launch request IDs were marked cancelled only while not
  RUNNING, with request locks held and provider absence checked again.
  No cancellation/teardown handlers were invoked for these queue records.
- Retired another 55 queued duplicate launch IDs after checking which
  original request each replacement controller had attached to. Twelve
  duplicates then RUNNING were deliberately not interrupted.
- Archived and removed over 500 obsolete or redundant replica records.
  Historical job records, cluster history and forensic backups remain.
- Deleted exactly one verified owned SUSPENDED Asia queued resource with no
  VM: `tpuswarm-v6e32-asia-q-9im1-7bfcb694-head-9uj83c1c-tpu-q`.
- Cleared obsolete current-worker bindings from queued/recovering jobs
  409, 410 and 456 using conditional updates. Their job IDs, queue entries,
  experiment state and recovery counts were not reset. Job 371 retained its
  RUNNING binding to v4-32 worker 94 / remote job 4.

## Controller Fixes

Implemented in the nested SkyPilot `sky/serve/replica_managers.py`:

- Recovering consolidated pool controllers attach to existing active launch
  request IDs rather than automatically submitting a second launch.
- A lost API stream reconnects to its existing request. Explicitly cancelled
  or interrupted requests can retry with a new ID while preserving the slice.
- Missing cluster handles no longer abort observation of subsequent workers.
- Missing launch logs no longer abort cleanup or truncate existing evidence.
- Pool probe observation errors return unknown, make the worker unavailable
  for new placement, and clear failure timers. They do not establish workload
  failure or justify teardown. Positive setup-job failure remains a failure.

All six local pool controllers were replaced, without restarting API PID
2318739, managed-job controllers, or TPU-host processes. Each recovered its
durable service version; target fields were updated without a rolling worker
replacement. This was not a training-code or model-config rollout.

The replacement supervisors initially inherited the operator shell's wider
CPU affinity. Final verification corrected every supervisor, descendant and
existing thread to CPUs 0-3. Future manual controller launches must use
`taskset -c 0-3` explicitly; copying environment sizing limits does not copy
affinity. Normal API-spawned recovery inherits the bounded API's affinity.

## Verification And Limits

27 focused tests passed in `tests/tpu_swarm/test_pool_reconciliation.py` and
the nested SkyPilot `test_pool_api_outage.py` and
`test_serve_replica_managers.py`. Both worktrees passed `git diff --check`.

At 04:25Z, API health/dashboard returned HTTP 200 in 4-7 ms. All four SQLite
databases accepted a BEGIN IMMEDIATE / ROLLBACK probe in 0.7-1.0 ms. The
current API log and all six replacement controller logs had zero
`database is locked` messages. This is a point-in-time check, not a guarantee
that SQLite on GPFS can never contend again.

All v4 and v5p provider identities in the initial inventory were preserved.
The v6e fleet continued changing during reconciliation, including SUSPENDING
resources and deleted/replaced east5-b requests. Do not attribute every such
transition to GCP preemption without audit-log evidence. Replica 193 was
initially still provisioning behind FAILED_CLEANUP; its reservation and VM
were subsequently both absent. Its local records were archived and removed
only after fresh full provider inventories confirmed that absence.

A responsive controller or completed launch is not proof of generation or
training. No successful new v4-64/v5p training step was established by this
operational work. Continue checking actual workload logs after allocation.

## Audit

Private audit directory under the durable SkyPilot HOME:

`.sky/recovery-audit/fleet-reconcile-20260909T040046Z/`

It contains before/after provider inventories, quota responses, per-pool
process/replica/config snapshots, archived records, cancelled request IDs,
controller logs, target changes, and affinity verification. Pickle/DB backups
may contain credentials; do not publish them.
