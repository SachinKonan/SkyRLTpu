# CPU placement seed regrade and v6e training settings

Completed 2026-09-17 on all eight hosts of v6e-32 worker 3744. These are
prepared training configurations; this investigation did not submit training.

Each run uses four physical trainer hosts (16 chips) and four inference hosts
(16 chips). The controller discovers a contiguous physical trainer block;
logical SkyPilot host order is not assumed to equal physical order.

| Model | Trainer TP / FSDP | Inference replicas / TP | Concurrent sequences per replica | Inference memory utilization | Trainer token budget |
| --- | --- | --- | ---: | ---: | ---: |
| Qwen3.5-27B | 8 / 2 | 4 / 4 | 16 | 0.80 | 45,056 |
| Muse-Glimmer-30B | 8 / 2 | 4 / 4 | 16 | 0.75 | 45,056 |
| Gemma4-31B | 4 / 4 | 4 / 4 | 16 | 0.80 | 90,112 |

Each model has one distributed trainer and one LoRA adapter, rank 32. All retain
learning rate 4e-5, full rematerialization, 16 groups x 32 rollouts, mean-baseline
advantages, importance-sampling loss, temperature 1.0, and 15 epochs. Muse
retains parameter host offload. Inference prefix caching is enabled and prefill
chunks are 1,024 tokens. Total context is 22,528; the phase-1 prompt plus thinking
ceiling is 16,384, leaving 6,144 tokens after that boundary. These are the existing
native-budget conventions, not 16,384 generated thinking tokens in addition to
the prompt. Trainer sequence buckets remain 18,432 and 22,528.

All eight hosts can grade: each case has 4 logical CPUs, 8 GiB aggregate RAM,
and CPU-only JAX. Sixteen case slots per host give 128 simultaneous cases per
slice, equivalent to 32 complete four-case submissions. Every slot has a disjoint
four-CPU affinity within CPUs 16-79 and a systemd memory/CPU/time envelope.
Ray reserves CPU, memory and a CPU grading token; it requests no TPU for grading.
Model RAM caches remain capped at 128 GiB per host. Candidate execution has a
180-second cap (170 seconds supplied to place), scoring 90 seconds, total 300.
The CPU prompt explicitly lists the libraries and shared resource limits.

## Seed results

Only previously valid, completed candidate records were included. No new model
generation or repair was performed. Each distinct source was regraded on all four
fixed Xplace-start cases using the production CPU task through Ray 2.58.0. New
scores and feedback were propagated to all matching source records before normal
PUCT admission: top two children per original root, followed by global code
deduplication. No TPU score remains in the imported candidates or best-value
statistics. Original generation roots and visit accounting are retained.

| Model | Distinct programs CPU-valid / regraded | Retained seeds / 32 | Best CPU reward |
| --- | ---: | ---: | ---: |
| Qwen | 78 / 79 | 18 / 32 | 0.439674169391 |
| Muse | 148 / 148 | 9 / 32 | 0.438879877453 |
| Gemma | 13 / 13 | 3 / 32 | 0.439773220753 |

The Xplace reference reward is 0.438879877453. Muse's best CPU score equals the
reference. Qwen's one failing program produces overlapping hard macros on ibm08.
The /32 denominator is pool admission capacity, not a validity denominator. The
same winning source can occupy several top-two selections and then be removed
by deduplication; this deliberately preserves the existing sampler behavior.
Gemma's source journal was incomplete (216 graded drafts); only its saved valid
programs were used. Muse's saved journal includes completed partial repairs;
ungraded generations were excluded.

## Validation and artifacts

- 960 seed-case executions plus 64 references (NumPy/JAX x four cases x eight
  hosts): 1,024 evaluations, 1,023 valid. All references passed.
- All eight hosts reached 16 concurrent candidate executions. Peak observed
  cgroup memory was 1.570 GiB; longest candidate wall time 178.756 seconds;
  longest completed trusted grading time 40.765 seconds. There were no OOM or
  timeout failures. This measures grading load, not simultaneous LLM training.
- 41 focused tests passed, plus 16 subtests. Old TPU profiles remain the default
  and retain their topology and budgets; CPU profiles opt in explicitly.
- Every regrade process and grading unit was verified stopped afterward.
- `settings.json` records the exact run settings; `execution-contract.json`
  records source hashes, input manifest, versions and limits.
- Each model directory contains the CPU-only PUCT seed pool, its checksum and
  compact results for every regraded source. New profiles pin these pool hashes.
- `runtime-summary.json`, `cleanup-audit.json`, and `full-focused-tests.txt`
  contain the corresponding evidence. Full per-case verdicts and logs remain in
  `.science/placement-cpu-20260917/host-*/regrade/`.

Run profiles: `tpu/swarm/ray_train/profiles/science-placement-v6e-{qwen,muse,gemma}-cpu-seeded-train-001.json`.
Prepared packages: `.science/deployment-science-placement-v6e-{qwen,muse,gemma}-cpu-seeded-train-001/`.
