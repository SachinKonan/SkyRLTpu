# Redesign: inference leases, open grading, unified discovery

Status: draft for review, 2026-10-14. Nothing here is deployed. The live RQ4 no-train runs build from the frozen
worktree `SkyRL-Main-infer-rq4-live` (1b8f4158), and no phase below touches it.

## Goals

1. Maximize inference, grading and training throughput across every TPU we hold.
2. Tolerate faults. A farm that dies has its work rerouted; a farm that comes back is noticed and used again.
3. Clean up reliably. Processes die with their job, on SkyPilot cancel or shutdown. Caches are reconciled and stale
   files deleted at startup.

Non-goals:
- Training throughput. The 2x v5p-8 trainer layout on the private v5p-64 works and stays as is.
- Changing grading semantics (time caps, scores) without a parity check.
- Routing by host type. Host type is recorded, not routed on, for now.

## Where we are (measured 2026-10-13)

Sources: `rq4-local/logs/perf-20261013/` (`bottleneck.py`, `ops.py`, `cg_summary.py`).

| Finding | Number |
|---|---|
| Share of each step where the farm's TPUs are idle while grades are still running | 76-98% (all runs except Qwen3.5 qubit) |
| Grading slots per farm, qubit / circuit | v5p-32: 40 / 160. v6e-8: 13 / 32 (14 / 36 after 1b8f4158) |
| Hours with a farm in the last 48 h, Gemma and Muse runs | 7-18 h |
| Grading thrown away per restart on long steps | 1.5-8 h (a restart redoes the whole step) |
| Inference: generation done after | 13-38 min of a 77-440 min step |

Grading is the bottleneck, and each farm can only grade for the one run that leases it. Farms of other models,
and farms between steps, sit idle while a short run waits.

### Two discovery services today

| | RQ4 discovery (della) | v5p-64 discovery (della) |
|---|---|---|
| Processes | `tpu.swarm.ray_train.borrowing_supervisor` (pid 2437495, since 10-08) and `tpu.swarm.ray_train.local_farms` (pid 2069063) | `SkyRLTpu-qwen-comparison/.science/qwen-comparison-20261006/od_watch.py` (pid 545072, since 10-10) |
| Finds farms by | `sky jobs queue_v2`, job name contains `inference-farm` (`borrowing_supervisor.py:101-128`) | Ops DB query `inference-farm%qwen38%` RUNNING, filtered by pool (`od_watch.py`) |
| Probes farms | `/health` and `/status` over SkyPilot SSH (`REMOTE`, `borrowing_supervisor.py:18`) | the same `REMOTE` snippet via `w.rpc` |
| Finds trainers | SkyPilot jobs in listed trainer pools that opt in (`lease_scope == 'run'`) | a hand-kept registration file (`discovery-targets.json`); trainers on the v5p-64 are reached by direct SSH to external IPs (`od_deploy.ssh`) |
| Publishes | pushes a URL list into each trainer's ingress (`POST target`); `local_farms` writes `rq4-local/farms.json` and keeps SSH tunnels for della clients | pushes candidates via `direct_discovery.tick`; spot farms are added only after the dedicated farm completed a request (`stable_completion`) |
| State | lock file `.rq4/supervisors/discovery.lock`, logs only | `discovery-targets.json`, `admission-state.json` and many audit files under `.science/` |

Both run the same probe code; they differ in how they find trainers and how they reach them. The v5p-64 path is a
set of one-off scripts outside the tree (`od_*.py`), and its launch path once skipped the placement environment
setup. That bypass produced zero-reward circuit grades (`QWEN38_V5P64_AUDIT_2026-10-12.md`, P0).

### Grading today

`tpu/swarm/ray_train/grading_service.py` already has most of what an open service needs:
- Submits are idempotent by `request_id` (`:224-228`).
- A bounded queue returns 429 when `running >= slots * queue_factor` (`:229-232`).
- Results are kept until fetched, with a long poll (`:269-283`).
- Infrastructure failures are kept apart from candidate failures (`:257-262`).

What stops it being open is the lease. Every entry stores its `lease_id`, and submit, result and cancel all require
the caller's `X-Lease-ID` (`serving.py:614-645`, `grading_service.py:225,275`). So only the run holding the
farm's inference lease can grade there.

## Design

### (a) Inference lease covers generation only (built: agent/req-protocol-inference)

**Lease** (farm side, one per farm; `serving.py`):

```
FREE --acquire(run)--> LEASED(run, adapter) --release | TTL expiry--> FREE
```

- One run, one adapter per farm. TTL (`external_pool_lease_seconds`) is renewed by the heartbeat; expiry is the
  only other transition. Release/expiry cancels the lease's unfinished groups.
- Fair queue: a run refused while the farm is leased is recorded; the next free lease goes to the run that has
  waited longest (a waiter silent for three heartbeats is dropped). Each run starts its farm walk at a different
  farm and retries with jitter.
- Client (`multi_borrowing.MultiRunBorrower`, the only multi-farm borrower): one admission lease before the first
  phase; inside a phase, leases follow outstanding groups (`demand()` = queued + running remotely, per-farm cap
  `external_pool_max_concurrent_requests`), capped by `external_pool_target_leases`; idle farms are released as
  soon as the phase has generated and nothing is outstanding (grading never needs a lease); all released at phase
  end and on every exit (runner `finally` closes the reservation on SIGTERM, error, or normal end). A farm that
  returns mid-phase is acquired as soon as demand needs it. Removed: run-scoped retention across steps, the fixed
  per-run `--farm-target` / `FARM_TARGET`.

**Generation group** (same request state machine as grading, `request_protocol.drive`):

```
submit --ACK(202, id)--> queued --> running --> done | failed | cancelled
   |  \--refused(429/503: full)--> reroute, not charged
   |   \--409 contract mismatch--> farm dropped, reroute (charged)
   poll misses x3 --discovery says farm healthy--> one more round (path, e.g. tunnel)
                  --otherwise--> lost: lease dropped, group regenerated elsewhere (charged)
   queued > 10 min while another farm is free --> cancel once (not awaited), reroute (not charged)
   unpolled for the request TTL (180 s) --> the farm cancels it
```

- Every group carries its contract: model, serving compatibility hash (includes `SKYRL_SWA_KV`), adapter,
  renderer date (`TTD_RENDERER_DATE`, recorded). The farm refuses a mismatch.
- Exactly once per request id: the farm hands a result over once; ids are fresh per attempt and abandoned ids
  are never polled again; duplicate submits of a known id return the existing view (counted).
- A group failing on farms 5 times (`external_pool_generation_max_failures`) is dropped and logged; the client
  counts it as a failed group and the step continues. Every regeneration is logged (group, farm, attempt) and the
  step metrics get `infra/groups_regenerated` and `infra/groups_dropped`.
- Journal (`generation-journal.jsonl` in the run dir) records every transition. A restarted client re-polls the
  groups its previous incarnation left on farms (run identity, not the lost lease), serves finished ones without
  regenerating them, and regenerates only the missing ones; reclaimed results nobody asks for are dropped and
  logged at phase end.

**Rollout:** farms must be rebuilt for `/skyrl/v1/generation/*`, the fair lease queue, contract checks and
generation TTL (older farms keep working: clients fall back to `/v1/completions` when a farm does not advertise
`generation_protocol`). Clients (local runner / trainers) need the new ingress; generation-scoped leasing is the
only multi-farm behaviour, so all runs on the new ingress release farms after generation.

### (b) Grading is open: admission with an ACK, no lease

Any client can submit to any farm's grading service, whatever model the farm serves.

1. **Admit or reject.** `POST /skyrl/v1/grading/submit` returns 202 plus a view (the ACK) only when the family's
   queue is under its threshold (`slots * queue_factor`, a per-farm setting). Otherwise it returns 429 with
   `retry_after` and current load, and the client tries another server. Nothing runs before the ACK.
2. **Identity without a lease.** The entry stores `owner_run` and a client token, issued in the 202 and sent with
   result and cancel calls. This replaces `lease_id` in `grading_service.py:225,232,275`. Request IDs stay 32-hex
   and idempotent. A resubmit with the same ID and token returns the existing entry.
3. **Choosing a server.** The client holds up to K servers at a time (K a client setting). Each grade goes to the
   less loaded of two random servers from the directory (`/grading/capacity` already reports
   running/queued/total per family).
4. **Fairness.** Each server keeps one FIFO queue per `owner_run` and serves the queues round-robin, so one run
   cannot take a whole farm. The farm's own inference-lease holder gets no priority.
5. **Failure and rerouting.** The client long-polls `result`. If the server stops answering for longer than the
   grade's time cap plus a margin, the client resubmits the same request ID elsewhere. That counts as an
   infrastructure retry, never as a candidate failure.
   - The existing `abort_training_step` after N infrastructure retries stays as the last resort.
   - A dead server's late results are ignored.
6. **Version check.** Requests carry `grader_sha256`, a hash of the grading code and inputs (the `tpu/science`
   grader files, case inputs, family contracts). A farm with a different hash rejects with 409. This is separate
   from the serving `compatibility_sha256`, which covers model and vLLM files and no grader file.
7. **Host metadata.** Every result records `host` (already in metrics), `host_class` (TPU type, e.g. v5p / v6e),
   `cpu_model` (e.g. Xeon 8481C / EPYC 9B14), the per-grade core and memory reservation, and `grader_sha256`. The
   client stores these with the trajectory record, in the same JSONL and cache entry as the reward. Routing ignores
   them for now. They are what later lets us check whether a score depends on the CPU it ran on.
8. **Results are cached with the trajectory.** A grade result is written once, next to its trajectory, keyed by
   request ID. A restarted client reloads finished grades instead of regrading them. This also covers "save
   finished grades so a restart resumes the step".

Why this is safe for time-capped graders: the per-grade core pinning and memory cap stay exactly as they are, so a
grade's own budget does not change with who sent it. The open question is CPU model (Xeon vs EPYC); see (f).

### (c) One discovery service

One process on della replaces `borrowing_supervisor`, `local_farms` and `od_watch`. It owns one directory file.

- **Sources.** SkyPilot managed jobs in our pools (farms and trainers), plus static entries for the private v5p-64
  partitions (host, external IP, role, port), declared in a checked-in file rather than `discovery-targets.json`.
- **Per farm:** model, serving hash, grader hash, grading capacity per family, inference lease holder, health,
  `last_seen`, `first_seen`, and an incarnation ID. The incarnation is the farm's start time, read from `/status`.
- **Heartbeat.** It probes every farm every 30 s. A farm is `unhealthy` after 3 missed probes and `gone` when its
  job is no longer RUNNING. A farm seen again with a new incarnation counts as come back. Its old leases and grading
  entries are void, and it rejoins as a fresh farm. Clients see this through the directory, not through errors.
- **Reaching farms from della.** Della cannot reach private TPU IPs, so the service keeps one SSH tunnel per farm
  (as `local_farms` does now) and publishes the tunnel URL next to the private URL. TPU-side clients use the private
  URL; della clients use the tunnel.
- **Pushes.** For trainers, it pushes the inference candidate list as today. For grading, clients read the
  directory themselves; no push is needed.
- **Never acquires leases or cancels jobs.** Arbitration stays in the farms.

### (d) Launcher for the private v5p-64

- A launcher in the tree turns a profile into the same run script SkyPilot would run, and starts it on named
  v5p-64 hosts over SSH. It uses `tpu/science/package_training.py` and `tpu/swarm/ray_train/build.py`.
- That replaces the `od_deploy*.py` / `od_prepare*.py` scripts. The point is that the v5p-64 runs the exact
  generated preparation, so the circuit environment gap from the audit cannot recur.
- Profiles name the partition (`hosts: ["v5p64-0", ...]`). The launcher registers the job in the discovery file
  and starts it under a systemd unit (see (e)).

### (e) Process and cache lifecycle

- **TPU hosts.** Each job runs everything (Ray, engines, ingress, graders) inside one systemd scope created by the
  job's run script. Cancel or shutdown stops the scope, which kills the whole cgroup. Grading units already run as
  systemd units; the leftover private Ray clusters and `ray::IDLE` processes are what this fixes.
- **Startup check on TPU hosts.** A job refuses to start only if a live scope of another job exists. Leftovers of
  dead jobs are stopped instead of failing the launch. This replaces "existing private executor Ray cluster;
  refusing to start", which today needs a manual `clean_worker.sh`.
- **Della.** The scheduler, discovery and each run client run as systemd user units. That isolates them from
  session-level kills like the 18:37Z mass kill on 10-13, and restart is then a unit restart, not
  `setsid nohup` plus a pid file.
- **Caches.** At startup, each host reconciles its caches (model weights, XLA compile cache, adapters, uploads,
  checkpoints) against a manifest of what the job needs. It deletes anything not listed and not in use, logging
  what it deleted.
  - The checkpoint rules from 19ec8c8d stay: keep the newest two, delete only after GCS verification.
  - Adapter archives of retired versions go after each swap (already in `serving.py`).

### (f) Smaller grading reservations, with a validation gate

Data so far: per-grade cgroup counters, 7 model/host/family combinations, about 19,800 grades in total
(`cg_summary.py`).

| Family | Reserved | Cores used, p50 / max | Peak memory, p90 / max |
|---|---|---|---|
| circuit (per case) | 4 cores, 4 GiB | 1.0 / 1.4 | 0.27 / 0.78 GiB |
| qubit (per program) | 10 cores, 20 GiB | 4.0 / 4.0 | 1.39 / 4.18 GiB |

**Proposal:**
- circuit: 2 cores, 1 GiB per case.
- qubit: 5 cores, 6 GiB per program.

The 4 GiB first tried for qubit is too tight: two programs peaked at 3.9 and 4.2 GiB. That roughly doubles circuit
slots and gives 2x qubit slots per host.

**Validated so far:** usage only. The score test (`perf-20261013/gradeab/`) is not finished.
- It has 23 of 28 rows of its first phase (qubit at today's reservation) and nothing from the smaller setting.
- Its v6e-8 lab worker was preempted 7 times. It now resumes from saved rows.

**Gate:** roll out only if scores, valid/invalid outcomes and cases completed match between the two settings within
the same-setting noise, on both v5p and v6e-8.

**Rollout constraints** (from `propagation-plan.md`):
- The qubit reservation is sent by the client and must match the farm's copy exactly. The client and farm change
  together, under a new contract version that farms accept alongside the old one during the switch.
- The circuit reservation is part of each run's saved bootstrap settings, so it needs new run IDs or an explicit
  exemption.

Free wins that need no gate are already done or listed:
- v6e-8 caps 13 -> 14 qubit and 32 -> 36 circuit (1b8f4158).
- v5p qubit admission by measured memory rather than the 20 GiB cap: up to 16 per host. This needs a code change.

## Migration (never touches the frozen live runs)

1. **Directory and lifecycle.** Build the unified discovery service, run it alongside the old ones and compare its
   directory with `farms.json` for a day. Add systemd scopes and startup cleanup to new farm builds only.
2. **Open grading.** Lease-free submit/result with client tokens, per-run fair queues, `grader_sha256`, host
   metadata in results and trajectories, and caching results with trajectories. Test on two new farms with a new
   no-train run that sends grading to both farms and to a farm of another model. Measure step time against the
   current runs.
3. **Inference lease per generation.** For training clients, measure adapter reload time and lease wait.
4. **v5p-64 launcher.** Relaunch one Qwen3.8 trainer partition through it after its current run checkpoints, with
   the user's go-ahead.
5. **Reservations.** Apply after the score test passes, with a new contract version, to new farms and new run IDs.
6. **Switch.** Move live runs over at a natural break (a run finishing, or a deliberate restart at a step boundary),
   then retire the old discovery processes and the frozen worktree.

## Decisions (user, 2026-10-14)

1. **Grade parity across CPUs:** yes, time-capped grades may run on either CPU model; the CPU is recorded with the
   result and cached with the trajectory, so parity can be checked afterwards.
2. **K and the queue threshold:** K = 4 servers per client, `queue_factor` = 1.5 to start.
3. **Priority:** none. Every run gets the same round-robin share on every server, including the farm's own
   inference-lease holder. (Priority would have been enforced in the server's per-run queues, by giving the
   holder's run a larger round-robin weight or a few reserved slots; not built.)
4. **v5p-64 sharing:** yes, the launcher checks and claims hosts before starting, and refuses if another
   user's processes hold them.
5. **Old checkpoint copies:** yes, cache reconcile also deletes other ranks' local checkpoint copies, under the same
   rules as the leader (in the ledger, not the newest two, verified in GCS).
