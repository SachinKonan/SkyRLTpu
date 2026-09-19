# Twelve v4-64 single-job GRPO runs

Prepared 2026-09-19. **Not submitted.** RG-LRU is deferred; no v5p or sidecar jobs.
`jobs.json` lists each profile, local package, and committed v4 runtime reference.

| Task | Qwen | Gemma | Muse |
| --- | --- | --- | --- |
| AC2 | prepared | prepared | prepared |
| Circle packing n=26 | prepared | prepared | prepared |
| Circuit: all 17 IBM cases, Xplace starts, C helper | prepared | prepared | prepared |
| Qubit: Q20 only, all 24 circuits | prepared | prepared | prepared |

Each run owns one v4-64 slice (eight hosts, four chips per host).
Bootstrap runs eight TP4 inference engines, with no trainer or optimizer started.
It issues at most 32 concurrent requests of 16 completions each, up to 1,024
persisted drafts, and stops issuing requests once 512 distinct valid programs
are available. In-flight groups finish; the highest-ranked 512 distinct valid
programs become seeds. If the draft cap comes first, use all distinct valid
programs available. Zero valid seeds is a failed bootstrap, not empty training.
A preemption before a response is persisted can require repeating that request;
the draft cap is on journaled candidate slots, not unobservable device work.

All requests use the same persisted initial parent and task prompt. No separate
invalid-only repair round. Generation responses and individual grades are
journaled before seed promotion. Valid states preserve each environment's code
format, construction, feedback, and maximize/minimize ranking convention. Seed
admission is global rather than top-two per parent: selected programs become
independent search roots. Subsequent GRPO updates keep the normal PUCT sibling
and buffer limits. Only a completed, hashed seed snapshot permits transition.

The existing controller then retires the four engines on the dynamically chosen
trainer block, verifies TPU release, and starts the trainer. The other four
engines keep their deployment identities and prefix caches. Training uses
16 groups × 32 completions, 15 optimizer iterations, mean-baseline GRPO,
importance_sampling loss, LoRA rank 32, and native prompt+thinking allowance
16,384 within context 22,528 (the existing answer allowance/safety margin).

| Setting | Qwen | Gemma | Muse |
| --- | --- | --- | --- |
| Learning rate | 1.5e-4 | 4e-5 | 4e-5 |
| Trainer TP / FSDP | 8 / 2 | 4 / 4 | 8 / 2 |
| Trainer token budget | 45,056 | 90,112 | 45,056 |
| Inference per host | TP4, max 16 sequences | TP4, max 16 sequences | TP4, max 16 sequences |
| Inference memory fraction | 0.80 | 0.80 | 0.75 |

Science grading runs on all eight hosts: 16 task slots per host, 4 CPU cores and
8 GiB per task (128 total slots). Circuit runs all 17 cases and optimizes their
unweighted mean proxy cost, requiring legality on every case. Q20 uses its
existing 24-case total SWAP objective. Math tasks retain their existing Ray
payload grading, with two CPU cores per task; no new science slot policy is
applied to math.

These are fresh bootstrap experiments with fresh model/optimizer state. They do
not import previous Q20 continuations. Each has a unique run directory and
compile-cache write prefix. Matching v4 runtime compile caches are read-only
seeds. Checkpoint recovery is enabled within each new run, with strict restore
and systemd runtime ownership; old jobs and checkpoints remain untouched.

Local packages live under `.science/packages/single-v4-20260919/<model>/<task>/`.
They contain a SkyPilot YAML and immutable code archive, but are not uploaded.
Rebuild math packages with `python -m tpu.swarm.ray_train.build PROFILE --output OUTPUT`;
rebuild science packages with `python -m tpu.science.package_training --profile PROFILE --output OUTPUT`.
See those modules' `--help` for their flag syntax before invocation.

Validation is local: config contracts, actual task prompt construction, bounded
bootstrap and interrupted-journal recovery, seed-pool loading, legacy bootstrap
compatibility, packaging, and existing science/lifecycle regression tests. The
new bounded bootstrap has not run on TPU hardware yet. Existing 16x32 science
bootstrap-to-training transitions are the foundation, not evidence of a completed
new 32x16 campaign.

Fleet snapshot during preparation: SkyPilot reported 11 ready v4-64 workers,
six assigned and five unassigned. Active scheduler records were AC2 1184/1185/1186
and Q20 1059/1061/1063. GCP separately listed 12 ready v4-64 slices and one creating;
provider readiness does not establish admission or runtime health. No cancellation
is required to prepare these configs. Prefer idle capacity for initial launches,
then reassess existing runs at durable checkpoint boundaries if capacity is needed.
