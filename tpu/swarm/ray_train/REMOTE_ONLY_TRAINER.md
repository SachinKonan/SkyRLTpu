# Remote-only trainers and farm-extended grading

A remote-only trainer trains on **every** host of its slice and runs **no**
local vLLM engine. All generation comes from one or more inference farms it
leases at once; candidate grading runs on its own hosts and, through the same
leases, on the farms' CPUs. The first profile is the four-host v4-32 Qwen AC2
pilot, `profiles/remote-only-v432-qwen-ac2-pilot-20260922.json` (TP8 x FSDP2
over sixteen chips, one optimizer step).

This document describes what differs from `HYBRID_INFERENCE.md` (one farm next
to local engines) and `INFERENCE_BORROWING.md` (the lease protocol), which
remain the reference for everything not mentioned here.

## Configuration

```json
"inference": {
  "remote_only": true,
  "external_pool_updates": true, "external_pool_lease_scope": "run",
  "external_pool_require_initial": true, "external_pool_scheduler": true,
  "external_pool_attestation": true,
  "external_pool_target_leases": 2, "external_pool_candidate_limit": 16,
  "external_pool_required_compatibility": [], "external_pool_queue_alert_seconds": 60,
  "request_timeout": 86400
},
"grading": {"families": {"ac2": {"slots_per_host": 16, "cpus": 2, "memory_gib": 4}},
            "farm_transport": true, "local_systemd": true}
```

* `remote_only` is accepted only for `trainer.hosts == hosts` on tpu-v4-32 with
  `tp * fsdp == 16`, run-scoped attested borrowing, no bootstrap, no arena or
  placement roles, and `request_timeout >= 86400` (requests queue inside the
  ingress until a farm is leased). The launcher disables every layer that could
  abandon a queued request: `SKYRL_EXTERNAL_WATCHDOG_{INFLIGHT_SEC,ABANDON_SEC,
  MAX_REDISPATCH}=0` (`max_redispatch <= 0` is unlimited) and
  `TTD_SAMPLING_PROGRESS_TIMEOUT=-1` (SDK stuck detection off). Profiles may
  not override these keys.
* `external_pool_target_leases` farms are held simultaneously; the supervisor
  offers up to `external_pool_candidate_limit` URLs (every farm the run
  already owns plus free compatible farms, v5p before v4-32). There is no
  fairness rule between runs: the farm's atomic acquire decides.
* `external_pool_required_compatibility` lists accepted serving-identity
  hashes (v5p and v4-32 farms can differ). Empty means the first attested
  farm's hash is learned and required from every later farm.

## Generation

`multi_borrowing.MultiRunBorrower` keeps one `RunBorrower` per candidate URL and
reconciles toward the target: acquisition is sequential (never more than the
target in flight or held), a farm serves only after the current adapter is
attested on it, and a farm re-acquired mid-phase republishes the adapter before
it serves again. Members use `request_failure_policy = 'probe'`: a failed
request pauses the lease and wakes the heartbeat, which re-attests; only lease
rejection (409/410), heartbeat/health-grace expiry or run close loses a lease.
Farm-side 4xx request rejections surface to the API server unchanged.

The hybrid scheduler runs one pool per eligible farm. Whole `n`-completion
groups go to the pool with the earliest predicted finish; a failed remote group
is re-queued at the head and retried on another farm (never locally, there is
no local engine). While no farm is eligible the queue waits without a deadline
and reports `remote_queue_waiting` every `external_pool_queue_alert_seconds`;
the controller likewise waits for the first reservation without a deadline
(`farm_admission_waiting`) and reports `remote_leases` / `remote_leases_zero`
without ever failing the run for a lack of farms.

The ingress on the trainer head is control-only: it stores adapters, owns the
leases, routes requests and reports `/status` (`remote_only`, `leases`) and
`/skyrl/v1/borrowing/services` (`target_leases`, `candidate_limit`,
`accepted_compatibility`). `/health` means the control plane is up; `/tokenize`
is refused (503). The controller verifies that Sky ranks follow the physical
z rows before starting the 1,1,4 trainer grid (`select_v4_32_topology.py`).

## Farm side: cancel on expiry

Abrupt trainer loss must not quarantine a farm. `farm_cancel_grace_seconds`
(default 5) after a lease expires with work in flight, the farm ingress
cancels the requests admitted under that lease, disconnects the engines' HTTP
requests (vLLM aborts on disconnect) and cancels any grading it dispatched,
then reports `lease_inflight_cancelled`. Quarantine remains the last resort
after `farm_drain_timeout` (600 in the relaunched farm profiles). A lost
request is re-queued by its owner; nothing enters a batch twice.

## Grading

Grading families (`ac2`, `routing`, `placement`) are declared per profile in
`grading.families`. On a farm they bound the Ray slot tokens
(`grading_<family>` per host), the Ray CPU budget (`17 + slots * cpus`) and
the CPU partition (`tpu/science/farm_resources.py`: science block on top, AC2
block below it, at least 24 service CPUs; engines and the ingress are pinned
to the service CPUs). On a trainer they bound the local pool the same way.

Farm endpoints, all fenced by the lease token (`X-Lease-ID`):

| Endpoint | Purpose |
|---|---|
| `POST /skyrl/v1/grading/submit` | `{request_id, task, owner_run, scope, spec}`; idempotent by `request_id`; 429 with `retry_after` under backpressure |
| `GET /skyrl/v1/grading/result/{id}?wait=` | long poll; `{state, result, error{class, detail}, host, metrics}`; 404 after retention |
| `POST /skyrl/v1/grading/cancel/{id}` | idempotent cancellation (Ray force-cancel; the unit's owner watchdog stops the candidate) |
| `GET /skyrl/v1/grading/capacity` | per-family slot totals, running/queued, `ready` after the startup reference grade |

The ingress never runs candidates. It dispatches the same Ray tasks trainer
hosts use: `tpu.science.ac2_grade.grade_ac2` (AC2, new), `ray_cpu.grade`
(routing) and `placement_ray.grade_cpu_case` (placement). AC2 candidates run
in a transient systemd unit (`MemoryMax`, `CPUQuota`, `AllowedCPUs`,
`RuntimeMaxSec = eval + 15`, `PrivateTmp`, `KillMode=control-group`) through
`tpu.science.ac2_runner`, which reproduces the discover sandbox semantics and
writes `started.json` before the candidate starts. The advertised 1000 s
budget and the 1100 s evaluator envelope are unchanged; admission waiting is
measured separately (`admission_wait_seconds`).

Trainer side, `tpu/science/grading_transport.py` grades through one scheduler
over the local Ray pool and one pool per eligible farm lease (listed by the
trainer ingress at `GET /skyrl/v1/grading/farms`). Each request keeps one
`request_id` across attempts; the first published result wins and later
duplicates are dropped. Failure classes are explicit:

* infrastructure (transport errors, 5xx, expired results, lease loss, Ray
  worker/node loss, admission timeout, unit died before `started.json`):
  retried on another pool up to `grading.max_infra_retries`, then
  `GradingInfrastructureError` (`abort_training_step`) aborts the step exactly
  like `ScienceInfrastructureError`;
* candidate (timeout, crash, exception, non-finite output): reward zero, as
  before.

AC2 uses `TTD_EVAL_BACKEND=hybrid` (`SandboxRewardEvaluator.run_transport`);
verifier injection and the driver-side re-verification stay on the trainer.
Identical candidate sources within one step are graded once
(`grading_dedup`, visible as `metrics.grading_dedup_hit`). Science graders take
the transport path whenever `SKYRL_GRADING_URL` is set and keep their
`ScienceInfrastructureError` contract.

Events in `inference-events.jsonl`: `grading_dispatched`, `grading_retry`,
`grading_infra_failure`, `grading_completed`, `grading_farms_changed`,
`grading_late_result_dropped`, `grading_no_pool` (trainer);
`grading_submitted`, `grading_finished`, `grading_cancelled`,
`grading_cancel_all`, `grading_readiness` (farm).

## Relaunching farms

Farm-side grading needs the new ingress code and slot tokens, so the Qwen farms
are relaunched from `profiles/farm-v432-qwen-grading-ac2-20260922.json` and
`profiles/farm-v5p32-qwen-grading-ac2-20260922.json` (AC2 family, drain 600,
cancel grace 5, reserve 192 GiB). A farm with a science family
(`farm-v432-qwen-grading-routing-example-20260922.json`) is packaged through
`tpu.science.package_training`, which prepares `.science/venv` on every host.
The supervisor still never provisions, cancels or replaces farm jobs.

## Pilot checklist

1. Inventory trainers, farms, queues and TPU VMs without cancelling anything;
   relaunch the two Qwen farms on the new bundle; on each farm host check
   `sudo -n systemd-run --version`, `ray status` (`grading_ac2: 16`, 49 CPUs),
   `/status.grading.ready == true` and the compatibility hash (list both hashes
   in `external_pool_required_compatibility` if they differ).
2. Trainer `/status`: `remote_only`, `borrowing.held` reaches 2,
   `capabilities.accepted_compatibility`; supervisor tick pushes up to 16 URLs.
3. Withdraw one farm mid-step: its groups complete once on the other farm
   (`hybrid_generated` count equals the batch), `grading_retry` events only.
4. Withdraw both farms: `remote_queue_waiting` / `remote_leases_zero` every
   60 s, API rows stay PENDING, relaunching a farm finishes the step.
5. Kill the trainer: farms report `lease_inflight_cancelled`, return to
   `unleased`, no quarantine, no relaunch.
6. Grading: `grading_dispatched` pools include `local` and both farms; parity
   of eight saved candidates between the `ray` and `hybrid` backends; a
   `kill -9` of a farm Ray worker removes its unit within a second and the
   trainer retries elsewhere; a candidate timeout stays reward zero.
7. One completed GRPO step, durable checkpoint, clean release on both farms.
