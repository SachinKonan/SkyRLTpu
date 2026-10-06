# Pool Submission Recovery, 2026-09-08

## Scope

This operational change preserves the existing SkyPilot API, workload
controllers, and TPU reservations. It is not a model/executor rollout.
Changes in other worktrees and unrelated dirty files are not included.

## Fixes

- `sky/provision/gcp/instance_utils.py` in the SkyPilot submodule now raises
  `ExecutionPausedError` for infinite queued-TPU waits in a scheduled API
  request. Newly created and adopted reservations are preserved. Resume uses
  the existing cluster-label adoption path, not a replacement queued resource.
  Finite deadlines and in-process callers retain their previous behavior.
- `sky/utils/controller_utils.py` separates consolidated pool admission from
  executor process concurrency. `SKYPILOT_POOL_MAX_INFLIGHT_LAUNCHES` bounds
  lightweight launch threads (default 256; deployment setting 160). Actual
  launch processes remain limited by the API configuration.
- Asia uses capacity-only pool version 8; Europe uses version 5. Targets remain
  40 and 2. East5b remains capacity-only version 6, target 32.
- Three specifically verified SUSPENDED Asia pool requests, each with owner
  label `skypilot-user=sk7524` and no TPU VM, were deleted and verified absent.
  No other user's resources or unrelated judge requests were deleted.

## Live Deployment Bridge

The original bounded API remains PID 945361 on port 46580. Its four existing
long executors were already inside blocking waits and have not been killed.
Their in-memory imports predate the pause patch.

`skypilot_queue_drain_worker.py` runs one supplemental process on two of the
API's four affinity CPUs. It processes existing pending/waiting pool launch
request IDs through SkyPilot's normal execution wrapper. The wrapper atomically
claims executable IDs and skips RUNNING/finished IDs. Cluster locks and cloud
identity checks still apply. It does not create its own launch request IDs.
Retryable requests remain durably WAITING and receive a retry delay. New IDs
are serviced before repeated attempts, avoiding starvation by old waits.

The first bridge used the shared executor machinery, which prefetched requests
into another FIFO. It was replaced with the serial bridge to avoid head-of-line
blocking by east5-b quota errors. Only that supplemental process was replaced;
its one interrupted east5-b launch ID was restored to PENDING after its process
was confirmed dead. The API executors and reservations were preserved.

Current bridge PID: 1337506. Log:
`/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-queue-drain-serial-20260908T193316Z.log`

Before changing the live bridge, inspect its current request and log. Do not
blindly kill an active worker or launch a second bridge. It takes a singleton
lock under the durable SkyPilot HOME and exits between requests when the API
process changes. On the next deliberately scheduled bounded API restart, new
executors import the native pause fix; the bridge should no longer be needed.

Five pool supervisors/controllers were recovered one at a time to apply the
admission fix: Asia, Europe, v5p, east5b, and v4-64. The v4-32 controller was
left running. Recovery can leave more than one launch request ID for a logical
worker; request counts must not be interpreted as unique GCP reservations.

Pre-change SQLite backups:
`/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/recovery-backups/queue-drain-20260908T192119Z/`

## Verified Limits And Remaining Checks

- GCP queued-request limits: east5-b 40, east5-a 50, Asia 48, Europe 50,
  central2-b 100. These are shared project limits, not per-user targets.
- At inspection, east5-b's 40 slots were occupied by other workloads, including
  unrelated jobs of this user. Their lifecycle is not owned by this repair.
- Actual new v5p requests were accepted through the serial bridge. Subsequent
  submissions hit `TPUV5PPreemptiblePerProjectPerZoneForTPUAPI`, limit 1536.
  That is a provider core-quota rejection, not an internal SkyPilot backlog.
- The 12 original v4-64 queued requests map to known SkyPilot cluster handles.
  Ten correspond to historical FAILED_CLEANUP workers 44 and 65-73; only two
  correspond to then-current workers 84 and 85. They were preserved, not
  deleted. Counting all 12 as usable current pool workers would be incorrect.
  Recovery/adoption of those historical worker records needs a separate,
  reservation-preserving reconciliation before any cleanup is considered.
- Submission remains asynchronous. Verify actual GCP queued resources and
  nodes, not just PROVISIONING records or configured targets, before claiming
  any pool is full. Shared quota may prevent our targets from being reached.

## Tests

56 SkyPilot tests passed across `test_gcp_queued_resources.py`,
`test_gcp_queued_resource_pause.py`, and `test_pool_inflight_launch_limit.py`.
Three repository tests passed for queue-bridge scope and capacity-only setup.
The live API health check remained responsive and the measured CPU cgroup
interval showed no new throttling after controller recovery.
