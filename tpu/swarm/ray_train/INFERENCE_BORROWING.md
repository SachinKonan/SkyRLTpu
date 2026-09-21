# Borrowing an external inference service

Optional consumer integration for single-model, single-adapter native-budget
training through the Ray Serve ingress. The default URL map is empty: existing
profiles keep their local inference path. No remote Ray connection is required.
This does not change GRPO/PWC, trajectory grouping, grading, or optimizer steps.

## Configuration

Merge these fields into a profile's `inference` object. Keys are **exact** base
model IDs, identical to `config.model` and the farm's `/v1/models` entry. A shared
map can contain Qwen, Gemma and Muse entries; a run only uses its own entry.

```json
{
  "routing": "ingress",
  "external_pool_urls": {
    "Qwen/Qwen3.5-27B": ["http://qwen-farm-1:8000", "http://qwen-farm-2:8000"]
  },
  "external_pool_updates": false,
  "external_pool_engines": 4,
  "external_pool_max_n": 1,
  "external_pool_max_concurrent_requests": 1,
  "external_pool_rpc_timeout": 3,
  "external_pool_prepare_timeout": 120,
  "external_pool_release_timeout": 15,
  "external_pool_lease_seconds": 300,
  "external_pool_heartbeat_seconds": 30,
  "external_pool_health_grace_seconds": 90
}
```

These are placeholder URLs, not deployed endpoints. Each URL addresses one farm
ingress controlling four engines. At most two candidate URLs may be listed per
model. A sampling phase borrows **at most one farm** alongside the job's existing
local engines. Requests are balanced by active requests per engine; each whole
`n`-completion request goes to one destination, preserving group construction.
The farm performs its own routing among its four engines.

**Batching/concurrency acceptance is explicit.** Defaults allow only `n=1` and
one simultaneous remote HTTP request, reflecting the reported sequential-only
validation. A normal `GROUP_SIZE=32` phase therefore stays local and does not
claim a farm with these defaults. After validating the farm at the intended
batch size, set `external_pool_max_n` to 32. Raise
`external_pool_max_concurrent_requests` separately after concurrent validation
(for example, 4 permits four simultaneous full-group requests). Exclusivity
does not establish numerical parity or concurrent generation correctness.

## Sampling lifecycle

1. At the pipelined sampling boundary, inspect the committed local adapter and
   try the two services. Health and base-model identity must match. An atomic
   lease claim is authoritative; an empty adapter list alone is not.
2. Upload the immutable committed adapter archive with its SHA-256. Require
   matching lease owner, adapter alias/hash and readiness on every remote engine
   before admitting remote generation. The local adapter publication barrier
   still covers only local engines. External preparation is bounded separately.
3. Route eligible requests to local and borrowed engines, retaining the exact
   native token/logprob/mask/audit response. Only the outgoing remote model alias
   changes. No per-token streaming or thinking-budget bypass is introduced.
4. Release after the sampling phase finishes, including its grading and streamed
   forward/backward work, before member finish/optimizer publication. We retain
   the existing sampling/training overlap rather than releasing on the first
   forward/backward call. Adapter replacement also closes a borrowed phase.
5. Base-model science bootstrap uses the same ownership lifecycle without a
   LoRA upload. Its batch/concurrency limits still apply.

The training client heartbeats its local ingress, which independently renews
the remote lease. If the client disappears, the local phase expires after 300s
by default, interrupts outstanding borrowed waits, and attempts release. If the
ingress disappears, remote lease expiry permits later ownership transfer. A
server can still need time to drain old admitted operations before reuse.

## Failure behavior

- Busy/offline farms at phase start: use another listed farm or stay local.
- A disconnected/erroring remote request: disable that farm for this phase and
  retry the unfinished request locally using the original adapter and payload.
- A renewed lease with the same owner/adapter but degraded engine health pauses
  **new** remote requests. Previously admitted requests may complete during a
  fixed 90-second health grace period. A healthy renewal resumes admission.
  Repeated degraded renewals do not restart the grace timer.
- Transport errors and temporary renewal responses (408, 429, 5xx) also pause
  admission and retry on the heartbeat interval. They never extend the last
  acknowledged lease deadline. In-flight work falls back when the first of
  health grace, acknowledged lease, or client phase expires. Successful degraded
  renewals extend the lease, but not the health grace. These are monotonic-clock
  deadlines; event-loop scheduling can delay detection during local overload.
- A confirmed owner/lease/adapter change, explicit renewal rejection, expired or
  draining remote lease, or malformed protocol response immediately invalidates
  the lease. Late renewal acknowledgements cannot revive an expired/lost lease.
- Release keeps admission closed even if a concurrent heartbeat recovers health.
- Other in-flight requests on that failed lease also retry locally. Successfully
  returned responses are kept. The original remote operation might still run;
  its abandoned result cannot enter the local training batch a second time.
- Initial lease/hash mismatch or incomplete readiness: no remote sampling; the
  grace period only protects requests admitted after full identity verification.
- An ambiguous **acquire** acknowledgement blocks further claims until the
  original farm acknowledges cancellation. Farms advertising `acquire_protocol=1`
  accept an `acquire_id` (UUID hex) and their `/status` `instance` as
  `farm_instance`, alongside `owner_run` on `/acquire_lease`.
  The borrower retries `POST /cancel_acquire` with those three identity fields
  at the next phase/reservation attempt. The farm prevents future grants for that
  ID and drains any already-granted work before acknowledging
  `{cancelled: true, acquire_id: ..., instance: ...}`. Cancellation is idempotent
  and can arrive before the delayed acquire. Tombstones remain for the ingress
  lifetime. A queued server acquire could execute late; waiting one TTL alone
  never permits a second claim. Lost cancellation replies are retried.
  Legacy farms, unreachable farms, and changed ingress identities retain the
  conservative block; they do not prevent local training.
- Run-scoped initial admission waits at most
  `external_pool_initial_wait_seconds` (default 300), then logs
  `farm_admission_local_fallback` and proceeds with local engines. This also
  bounds profiles with `external_pool_require_initial=true`: that setting now
  means prefer an initial reservation within the admission window. Later phases
  can reconcile the pending acquire and borrow again. The deadline is enforced
  between bounded RPCs; it is not a real-time scheduling guarantee.
- Unconfirmed release stops renewal and pauses subsequent acquisitions through
  a conservative expiry interval. Cleanup calls include phase identity, so a
  stale client cannot release a newer local phase.
- Local inference failure retains existing job error/recovery behavior. External
  borrowing cannot make the local service itself fault tolerant.

## Required serving-side contract

This matches the lease API in `SkyRLTpu-science-multi-lora` (`LORA_FARM.md`):

| Operation | API |
| --- | --- |
| Claim | `POST /acquire_lease`: `owner_run`, `ttl_seconds` |
| Renew | Same endpoint plus the returned `lease_id` |
| Upload | `POST /skyrl/v1/upload_lora_adapter?lora_name=borrow-<lease_id>`; headers `X-Lease-ID`, `X-Adapter-SHA256`; tar body |
| Verify | `GET /status`: owner/lease, adapter hash/name, `ready_engines`, `expected_engines`, state |
| Generate | `POST /v1/completions` with `X-Lease-ID` and the verified model alias |
| Release | `POST /release_lease`: `lease_id`; require `state=unleased`, `released=true` |

The farm must fence admission by lease, independently verify loaded adapter
hashes, and drain actual operations during ownership handoff, including requests
whose HTTP caller disconnected. Release quarantines the old adapter; physical
unload and an empty `/v1/models` list are not required. A plain v2 ingress without
this lease contract is ineligible. The farm must also use the matching model
revision, tokenizer, adapter settings and native-budget runtime; the current API
does not attest the complete runtime configuration. Use trusted private endpoints.

Local control endpoints are `/skyrl/v1/borrowing/{begin,heartbeat,end}`. Status is
included under `borrowing` in local `/status`; events use the `borrow_` prefix in
`inference-events.jsonl`. Lease tokens and full transport exceptions are excluded
from these events. The packaging overlay includes the client phase hook.
`borrow_health_paused` records a safe reason and, for degraded acknowledgements,
state and ready/expected engine counts. `borrow_health_recovered` records resumed
health. Neither event includes a lease token or raw response body.

## Validation scope

Local HTTP/ASGI fault-injection tests exercise ownership conflicts, lost acquire
acknowledgements, hash mismatch, remote request/heartbeat failure, client expiry,
cancellation, draining, stale cleanup, disabled configuration, and bundle overlay
installation. Existing serving tests cover unchanged local adapter handling.
These checks do not establish live TPU throughput or numerical parity. No farm
was claimed and no running experiment changed during implementation.

The September 20 heartbeat correction additionally tests transient degraded
status, 503s and transport timeouts, successful in-flight completion during grace,
recovery, persistent degradation, identity changes during grace, acknowledged
lease expiry, late renewal and release/recovery races. It is a consumer-only
change; the existing farm API is unchanged. Running bundles do not hot-reload it.

## Controller-local endpoint supervisor

For spot workers whose addresses change, set `external_pool_updates: true` in
each participating training profile. `external_pool_urls` may be empty. This
creates the borrower and client sampling hook even before endpoints are known.
The default is false; static profiles retain their existing behavior.

Run one `borrowing_supervisor` process on the controller node with its existing
SkyPilot environment and SSH configs. It reads current RUNNING managed jobs
whose names contain `inference-farm` (case-insensitive), across all pools visible
to that SkyPilot API server. An optional `--farm-pool` additionally includes
legacy farms without the naming convention. It probes `/health`, the lease-capable `/status`, and the
model list, and obtains each farm's private address from its VM metadata. It
then updates only explicitly listed training job IDs or opted-in jobs in listed
trainer pools with matching model
URLs. It never claims a lease, uploads an adapter, launches a job, or resizes a
pool. It uses SSH for localhost HTTP calls because the controller may not have
a direct route to the private TPU addresses. Borrowers independently check
connectivity from their own hosts before acquisition.

Using the SkyPilot Python environment, first run a read-only pass:

```bash
python -m tpu.swarm.ray_train.borrowing_supervisor \
  --farm-name-contains inference-farm \
  --farm-pool tpuswarm-v4-32-central2-smoke \
  --trainer-job-id TRAINER_JOB_ID \
  --ssh-config-dir /path/to/sky-home/.sky/generated/ssh \
  --once --dry-run
```

For discovery by name only, omit `--farm-pool`. Job names select candidates;
they do not bypass health, four-engine readiness, exclusive leases, or the
borrower's model/runtime/adapter compatibility checks. v4-32 and v5p-32 farms
can coexist. New capacity does not preempt a healthy existing reservation.
The current borrower reserves one farm per training run, not every discovered
farm simultaneously. See [v5p farm profiles and submission](../../../docs/inference-farms-v5p32.md).

Replace `TRAINER_JOB_ID` with the intended job ID and repeat the flag for other
authorized training jobs. To keep updating in the background, use the same
command without `--once --dry-run`, with `nohup` and log redirection. The default
interval is 30 seconds after each pass. A per-login file lock prevents competing
supervisors on this node. Stop it with SIGTERM; training continues locally or
with the last known candidate list, subject to the normal acquisition checks.

The trainer's new `GET/POST /skyrl/v1/borrowing/services` endpoint is opt-in.
GET supplies model, run ID, service instance ID and current candidate URLs. POST
must match all target identity fields and replaces the list atomically. The
supervisor also requires the managed job name to match the run ID, preventing
a reused worker from receiving an update intended for an older job. These are
identity checks, not public endpoint authentication; use the existing trusted
private control network/SSH access. URLs use the same origin validation and
two-farms-per-model limit as static configuration.

List updates never modify an active lease or redirect an in-flight request.
The next sampling-phase acquisition uses a snapshot of the current list.
Unavailable farms are removed on a successful discovery pass. A controller
inventory error leaves lists unchanged, and borrowers still verify every
acquisition. No registry service, cloud object store, farm publisher, or new
heartbeat protocol is needed. Older running bundles lack this endpoint and
must be repackaged before they can participate; the supervisor reports them
as unsupported instead of modifying their processes.

Validation on September 20: a live `--once --dry-run` found six healthy farms
(two each for Qwen, Gemma and Muse) at their current addresses, and skipped
target job 1270 because it was no longer RUNNING. It made no HTTP writes.
The supervisor, borrower, ingress, package, command and serving checks passed
110 tests. The daemon has not been started against a newly packaged target yet.

### Local failure policy for the supervised trial

Borrowed-farm failure is recoverable only while local inference remains healthy.
With `inference.restart_limit=0`, the controller pins the local ingress instance
once startup succeeds and probes `/status` during execution. Three consecutive
transport/5xx probe failures fail the run; a valid probe resets the count.
A replaced ingress, exhausted engine restart budget, malformed status, or recorded
local engine failure fails the run immediately and enters owned-process cleanup.
Generation responses 400/404/413/422/429 are propagated as request errors without
poisoning the farm catalog; engine transport/5xx failures remain fatal under the
strict policy. It does not borrow a farm
to recover local inference. `/status` stays available during normal adapter
updates; `/health` returning 503 during an update is not used as a fatal signal.
Intentional bootstrap topology changes suspend this monitor until readiness.

The supervised Qwen trial also sets `max_restarts_on_errors=0`, so SkyPilot does
not repeatedly relaunch application failures. Provider preemption remains a
separate job-recovery mechanism. Already durable checkpoints are preserved.
Remote health grace is unchanged: pause new remote admission for up to 90 seconds
while preserving pending work, then fall back locally if it cannot recover.
Retries regenerate a whole unfinished request group; partial KV state is lost.
