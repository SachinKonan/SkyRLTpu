# Existing V5p Worker Recovery, 2026-09-08

## Cause

Workers 189, 197, 198, and 199 in `tpuswarm-v5p32-east5a-erdos` had
`sky_launch_status=FAILED` and `sky_down_status=FAILED`, no first-ready time,
and no application failure. Their logs show an API connection failure being
wrapped as the generic local-API-start-disabled RuntimeError. The existing
transport-interruption classifier missed that wrapper and attempted teardown.
Teardown also failed during the API outage/timeouts (plus SQLite locking for
189), leaving live GCP resources behind terminal FAILED_CLEANUP records.
Controller startup recovery does not retry that terminal status.

The fix in the SkyPilot submodule preserves ApiServerConnectionError when
local API auto-start is disabled. The replica interruption classifier also
recognizes older RuntimeError wrappers through their exception chain.
Authentication checks and the local-auto-start prohibition remain enabled.

## Recovery Operation

`recover_v5p_existing_workers.py` is a deliberately scoped repair, not a
general-purpose import command. Its allowlist contains only 189, 197, and 198
and their exact original TPU node names. Worker 199 was excluded when GCP
initiated suspension; its VM disappeared during the investigation.

Run with the existing durable API environment, including
`SKYPILOT_USER_ID=7bfcb694`, matching gcloud/ADC service-account identity,
`SKYPILOT_DISABLE_LOCAL_API_SERVER=1`, and single-thread BLAS/OpenMP limits.
Use the multihost `third_party/TPUSwarm/.venv/bin/python` interpreter.

```bash
python tpu/swarm/recover_v5p_existing_workers.py 189 --ssh-key /path/to/sky-key
# After inspection, add --run to bootstrap and re-enable readiness probing.
```

The operation:

- Authenticates gcloud and ADC, verifies original node and queued-resource
  identities, ACTIVE/READY/HEALTHY status, all four hosts, device idleness,
  disabled autostop, and no pending/running/waiting API operation on the worker.
- Keeps the terminal pool record unchanged during bootstrap. Runs SkyPilot's
  synchronous execution path in one bounded local process; no API restart or
  extra TPU launch is required.
- Substitutes an existing-only provision record and refuses resource creation,
  replacement, stop, deletion, and teardown. A refusal bypasses provisioning
  failure handlers, leaving the original reservation untouched.
- Requires bootstrap job 1 to succeed and rechecks original node and queue
  creation timestamps. Uses compare-and-swap on the exact original replica
  record to clear failed lifecycle flags. It does NOT set readiness true.
- Lets the existing pool controller independently probe job 1 and mark READY.

This tool is not idempotent for already-ready workers: it refuses them. A
failed bootstrap stays unschedulable for inspection; do not force its status.
Audit snapshots live under the durable HOME's `.sky/recovery-audit/`.

## Verified Outcome

189 was the canary: bootstrap job 1 succeeded, the normal controller marked
READY, and a queued workload was assigned. 197 and 198 were then recovered
serially with the same checks. Final provider checks confirmed all three
original node and queued-resource creation timestamps unchanged, with nodes
READY/HEALTHY and reservations ACTIVE. Pool status changed from zero ready
workers to three. The target remains 48.

Only the v5p pool supervisor/controller was recycled to load the outage fix.
Its process tree was identified and backed up, the supervisor suspended, and
the local processes killed without executing destructive Python shutdown
cleanup. Normal HA recovery started supervisor PID 2614283 on port 20001.
The shared API remained PID 945361 and responsive; its request executors were
not killed. Other pool controllers were not reloaded by this repair.
Do not copy that process operation without fresh PID, ownership, state, and
recovery-script checks. Ordinary pool down or graceful service shutdown is
not equivalent: it can delete workers.

Controller-reload backup:
`~/.sky/recovery-audit/controller-reload-1788907952186009133/`
under the durable API HOME, not the login user's HOME.

There is still a separate dispatch bottleneck. At final inspection each of the
three recovered workers had a PENDING sky.exec request. These are real pool
assignments, not proof of training. The shared API's original four long workers
remain occupied by provisioning. The revised eight-worker launcher settings
are not live, and this repair does not claim to fix global API dispatch.
Likewise the source fix is not automatically loaded into other existing local
controller/API processes until they are deliberately replaced.

## Tests

27 focused tests passed: 19 SkyPilot outage/replica-manager tests and eight
existing-worker recovery tests. They cover typed and wrapped API outages,
no teardown on outage, rejection of wrong owner/unhealthy or suspending
resources, mutation guards, identity changes, compare-and-swap, and normal
readiness probing after activation. No commits or pushes were made.
