# Farm-assisted v6e training

This opt-in path keeps a v6e run's four local TP4 inference engines and adds one
exclusive v4-32 farm with four TP4 engines. It serves one policy, with identical
adapter archives on both sides. It does not implement multi-policy training.
Approved regions are **us-east5-b and us-central1-b**. AC2 for Qwen3.5-27B,
Muse-Glimmer-30B and Gemma4-31B precedes qubit routing.

## Admission and ownership

Enable `external_pool_updates`, `external_pool_lease_scope="run"`,
`external_pool_require_initial`, `external_pool_scheduler`, and
`external_pool_attestation`. Configure `external_pool_max_n=32` and
`external_pool_max_concurrent_requests=4`. Existing profiles keep phase-scoped
borrowing unless explicitly opted in.

The controller starts local inference first and waits for a farm reservation
before starting trainers. The supervised discovery process only updates an
explicit set of pools/jobs. It preserves existing lease owners, prioritizes AC2,
and does not issue new qubit reservations while AC2 is outstanding. Farm claims
are atomic; discovery assignments are advisory. A recovering job can therefore
wait for the previous owner's lease to expire without stealing its engines.

The controller renews run liveness throughout optimization and checkpointing.
Sampling-phase end drains generation while retaining ownership. At the next
phase, remote publication occurs asynchronously; local generation may proceed.
Every new remote alias contains the lease ID and full archive SHA-256. All four
engines must acknowledge that hash before the remote route opens.

Attestation compares model revision, model/tokenizer metadata, serving wrappers,
installed package versions and Python source hashes. New farm claims must carry
the matching contract hash. This is stronger than `/v1/models`, but is not a
numerical-equivalence test. Hardware-specific cache flags are deliberately not
part of the shared contract: prefix caching stays **on v6e, off v4**.

## Sampling and failures

The initial recipe retains 16 groups of 32 completions (512 rollouts), original
thinking/context budgets, loss, seed, parent pool, masks and logprobs. Each HTTP
request stays whole. No regrouping, chunking, or cross-policy importance weights
are introduced. Four local and at most four remote groups execute at once;
remaining groups wait in one ingress queue. Dispatch uses measured completion
rates and avoids a remote dispatch when waiting for a local engine predicts an
earlier finish. These estimates require TPU measurement before claiming speedup.

A lost remote request is discarded and retried locally with the same payload and
adapter. Successful responses are returned once. Remote work may continue after
a disconnection, but its result cannot also enter the learner batch. Lease and
health deadlines stop acceptance of late results. Defaults: 300-second lease,
30-second renewal, 90-second health grace. An expired lease with stuck operations
is quarantined after another 120 seconds; its owning controller fails and normal
managed-job recovery reconstructs the farm. Strict local-engine failures still
fail the run. Publication failure never admits partially loaded remote weights.

## Rollout and validation

`../../science/results/hybrid-migration-20260920/prepare.py` builds separate
2-step AC2 canaries and one farm upgrade per model. It verifies the previously
recorded seed-file and canonical-pool hashes. Production checkpoint namespaces
are not reused by canaries. Qwen and Gemma target east5-b; Muse targets central1-b.
Both regions are accepted by the supervisor. Model/HF and Orbax caches remain
shared read sources; each new run writes to its own compile-cache prefix, seeded
from existing compile caches. TPU-local cache restoration uses the existing
verified GCS/tmpfs cache path.

Before replacing a farm, recheck its exact job, live lease, active count and TPU
mapping. Upgrade an idle farm only. Do not cancel running training to place a
canary. Immutable bundle SHA-256 and job IDs belong in the launch receipt.

Required gates before production cutover:

1. Two completed optimizer cycles, corresponding native responses, all-engine
   adapter reload, and durable checkpoints for each model. Inspect train-error
   metrics; job status or exit code alone is insufficient.
2. Resume a canary checkpoint without resetting weights, optimizer, parent tree
   or step. Existing runs move only from verified durable checkpoints.
3. Exercise farm loss and confirm whole-group local fallback without duplicate
   learner inputs; demonstrate quarantine/recovery for an abandoned request.
4. Run `python -m tpu.swarm.ray_train.hybrid_benchmark` on an idle ingress with its
   training client paused and controller heartbeat still running. Supply the
   committed adapter alias and 16 real tokenized request payloads. The tool warms
   both routes, runs three paired AB/BA rounds, verifies native responses, saves
   every answer, and requires actual remote completions in hybrid rounds.
5. Require at least 10% lower median generation wall time and no full-step
   regression. The benchmark's generation gate explicitly does not establish
   the full-step gate. Preserve token counts and response-length differences.

The `X-SkyRL-Local-Only: 1` header selects the same bounded local scheduler for
paired measurement without changing the sampling payload. No tuning of group
size, max sequences, or memory utilization precedes the baseline gate.

## Managed discovery

`borrowing_service.py` renders a user-systemd service with restart-on-failure,
a distinct singleton lock, and `--run-scoped-only`. Run it from an immutable
copy with the existing SkyPilot Python and an explicit environment file. Include
both `tpuswarm-v6e32-east5b-qwen35` and `tpuswarm-v6e32-central1b`, with farm pool
`tpuswarm-v4-32-central2-smoke`. Existing phase-based trial supervisors and their
jobs are not updated by this service. An inventory outage leaves previous URL
lists intact; independent lease checks continue fencing requests.

Rollback is to stop assigning farms to new jobs and launch/resume the prior
local-only profile from its durable checkpoint. Removing a URL from discovery
alone does not revoke an active lease. Never reuse a farm with abandoned work
until it drains or its owned runtime has been recycled.
