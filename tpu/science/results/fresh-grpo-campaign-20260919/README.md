# Fresh GRPO campaign: v4-64 and v5p-32

Authorized 2026-09-19. Fifteen logical experiments, each with two hardware
profiles (30 configs). Launch one hardware variant per experiment unless a
duplicate is explicitly desired. The initial deployment uses v4-64; v5p is
prepared as an alternative. RG-LRU was added after the initial twelve launches;
its six profiles are prepared and packaged, but have not been submitted.

| RG-LRU model | v4-64 profile | v5p-32 profile |
|---|---|---|
| Qwen | [Config](../../../swarm/ray_train/profiles/fresh-v4-qwen-rglru-grpo-lr15e4-s1-20260919.json) | [Config](../../../swarm/ray_train/profiles/fresh-v5p-qwen-rglru-grpo-lr15e4-s1-20260919.json) |
| Gemma | [Config](../../../swarm/ray_train/profiles/fresh-v4-gemma-rglru-grpo-lr4e5-s1-20260919.json) | [Config](../../../swarm/ray_train/profiles/fresh-v5p-gemma-rglru-grpo-lr4e5-s1-20260919.json) |
| Muse | [Config](../../../swarm/ray_train/profiles/fresh-v4-muse-rglru-grpo-lr4e5-s1-20260919.json) | [Config](../../../swarm/ray_train/profiles/fresh-v5p-muse-rglru-grpo-lr4e5-s1-20260919.json) |

Every run starts from the pretrained model and a fresh optimizer. No previous
bootstrap, discovered-program pool, or optimizer checkpoint is imported.
The task's ordinary reference program remains part of its initial prompt.
Recovery resumes only this new run's own journal/checkpoints.

## Experiments and priorities

| Task | Qwen | Gemma | Muse | Queue priority | Evaluation |
|---|---|---|---|---:|---|
| AC2 | 1 run | 1 run | 1 run | 100 (high) | Existing AC2 environment/reward |
| Circle packing n=26 | 1 run | 1 run | 1 run | 100 (high) | Existing n=26 constraints; maximize sum of radii |
| Qubit | 1 run | 1 run | 1 run | 50 | All 72 cases; one policy across Q20, Willow, Heron |
| Circuit | 1 run | 1 run | 1 run | 50 | All 17 IBM benchmarks; zero overlaps on every case |
| RG-LRU | 1 run | 1 run | 1 run | 50 | Pallas kernel forward/backward correctness and speedup on the pinned shape suite |

Submit all six math jobs before the six science jobs. These numeric priorities
control queue ordering; they do not preempt unrelated running jobs.
RG-LRU has its own explicit submission stage after these tasks.

Qubit: weighted added-CNOT costs, weights Q20=0.2, Willow=0.4, Heron=0.4 per
case. Reward B/(B+C), B=119874, equivalent to applying the same ratio to
weighted averages. All cases must pass; invalid submissions receive zero.
The model receives all 72 case counts, SABRE baselines and differences, plus
per-topology totals. This is not Q20-only. Case counts support separate
comparisons against published SABRE, LightSABRE and SimpleTES results.

RG-LRU: implement the existing `kernel(x, a, reset)` Pallas contract, including
forward and backward. Raw score is the geometric mean speedup over the fastest
calibrated honest baseline per shape (RecurrentGemma scan or XLA associative
scan), including the ragged holdout. Higher is better; raw speedup above 1 is
faster. Training reward retains the existing noise-floor adjustment and invalid
kernels receive zero. Grader infrastructure failures abort the rollout rather
than masquerading as candidate failures. TPU generations are separate benchmark
conditions; do not compare v4 and v5p raw latency as matched hardware results.

Circuit: unweighted arithmetic mean of proxy cost across all 17 cases,
reward 1/(1+mean_proxy_cost); any missing/illegal case receives zero. Xplace
initial layouts and the built-in fast C scoring helper are supplied. Feedback
includes each netlist's cost components, legality, candidate runtime/budget,
and separate grading time. Scores are not competition-judge verification.

## Bootstrap and training

| Phase | v4-64 (8 hosts, 4 chips/host) | v5p-32 (4 hosts, 4 chips/host) |
|---|---|---|
| Bootstrap inference | 8 TP4 engines | 4 TP4 engines |
| Training | 4 trainer hosts + 4 TP4 inference engines | 1 trainer host + 3 TP4 inference engines |
| Science CPU grading | 16 slots/host, 128 total | 16 slots/host, 64 total |

RG-LRU is the exception because its kernel grader needs TPU chips throughout:

| RG-LRU phase | v4-64 | v5p-32 |
|---|---|---|
| Bootstrap | 7 TP4 inference hosts + 1 grader host | 3 TP4 inference hosts + 1 grader host |
| Training | 4 trainer hosts + 3 TP4 inference hosts + 1 grader host | 1 trainer host + 2 TP4 inference hosts + 1 grader host |
| Grader concurrency | 4 independent one-chip case tasks | 4 independent one-chip case tasks |

Logical rank 1 is reserved for grading. On v4, physical topology is probed and
the four-host trainer row that excludes rank 1 is selected; the trainer leader
need not be rank 0. The client stays on rank 0, which remains an inference or
trainer host. The grader is never included in the bootstrap inference deployment.
On v5p, trainer rank 0 and inference ranks 2/3 retain the existing host layout.
Both use workload Ray v2 grader tasks, with no separate queue host or external
ARENA_QUEUE_URL. Startup runs grader and client-transport self-tests before
generation. Kernel grading uses the pinned JAX 0.10.2 environment, 20 timing
pairs, 180 s compilation and 900 s grading budgets per case; end-to-end waiting
retains the existing 14400 s allowance. Each case reserves 2 host CPUs and one
exclusive TPU chip. Candidate children retain the existing 64 GiB RLIMIT;
this is not a new hard cgroup memory reservation. Grader caches are scoped to
the run ID so v4 and v5p never share a write destination.

Bootstrap has no gradient updates. Up to 32 concurrent requests, each with 16
completions, share the initial task parent. Generate at most 1024 journaled
drafts; stop issuing requests when 512 distinct valid programs are available.
In-flight requests finish. Select the highest scoring 512 distinct valid
programs, or all valid programs if the draft cap is reached first. Zero valid
programs stops the run. There is no separate invalid-only repair stage.
Selected seeds become independent PUCT roots; the ordinary top-two-per-parent
admission rule applies to subsequent training, not this initial promotion.

After the seed journal and pool are durably saved, retire only the inference
engines on trainer hosts, verify TPU release, and start the trainer. Surviving
inference engines retain their identities/caches. GRPO then uses 16 groups x
32 completions per iteration for an initial 15 optimizer steps. Mean-baseline
advantages, importance_sampling loss, LoRA rank 32, seed 1. Native
prompt-plus-thinking allowance is 16384 tokens, total context 22528, leaving
approximately 6144 answer tokens less the existing safety margin.

| Model | LR | v4 TP/FSDP | v4 trainer token budget | v5p TP/FSDP | v5p trainer token budget | Inference memory fraction |
|---|---:|---|---:|---|---:|---:|
| Qwen 3.5 27B | 1.5e-4 | 8/2 | 45056 | 1/4 | 90112 | 0.80 |
| Gemma 4 31B | 4e-5 | 4/4 | 90112 | 4/1 | 22528 | 0.80 |
| Muse Glimmer 30B | 4e-5 | 8/2 | 45056 | 1/4 | 90112 | 0.75 |

Each inference engine is TP4 with max 16 sequences and prefix caching.
Muse's v5p variant deliberately uses one TP4 engine per host, instead of the
older launcher's two TP2 engines per host, to implement the requested four-
engine bootstrap and three-engine training phase. It does not read TP2 compile
caches. Other v5p trainer meshes follow the native launcher, including Gemma's
256 backward blocks. v4 Qwen retains the required ragged convolution path.

Science grading uses 4 CPU cores and 8 GiB per task, CPU-only, on every host.
Circuit schedules one netlist per task (180 s candidate budget, 180 s grading,
390 s worker envelope); qubit runs the full suite per task (1800 s total,
900 s build/solve and 900 s verification). Math retains the existing two-CPU
Ray grader; science's per-host slot admission is not imposed on math.
Compile RAM caches retain their 128 GiB limits. Every new run has independent
checkpoint and compile-cache write destinations. Existing matching compile
caches may seed compilation; they do not import trained weights or seeds.

## Files and submission path

Worktree: `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-science-placement`.
The exact 30-profile index is [jobs.json](jobs.json). Profile filenames:

```
tpu/swarm/ray_train/profiles/fresh-{v4|v5p}-{qwen|gemma|muse}-{ac2|cp26|qubit|circuit|rglru}-grpo-{lr15e4|lr4e5}-s1-20260919.json
```

Qwen uses `lr15e4`; Gemma and Muse use `lr4e5`. Packages and individual
submission receipts live under:

```
.science/packages/fresh-grpo-20260919/{v4|v5p}/{model}/{task}/
```

Build all profiles and immutable packages from this worktree:

```bash
export PYTHONPATH=third_party/discover:.:tpu
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
/scratch/gpfs/ZHUANGL/sk7524/tinker-cookbook/.venv/bin/python \
  tpu/science/results/fresh-grpo-campaign-20260919/prepare.py
```

Do not rebuild already submitted run IDs and use them as changed experiments.
Their recovery contract pins configuration and bootstrap implementation.
To build only the six RG-LRU additions, append `--task rglru` to `prepare.py`.
The existing indexed packages and submission receipts are preserved.

After matching gcloud/ADC identity, TPU/storage read probes, and fleet inventory:

```bash
export CLOUDSDK_CONFIG=/home/sk7524/.config/gcloud-tpuswarm-compute-sa-v6e32
export GOOGLE_APPLICATION_CREDENTIALS=/home/sk7524/.config/gcloud/vision-mix-compute-sa-key.json
export SKY_API_SERVER_URL=http://127.0.0.1:46580
export SKYPILOT_API_SERVER_ENDPOINT=http://127.0.0.1:46580
export SKYPILOT_DISABLE_LOCAL_API_SERVER=1
export SKYPILOT_CONFIG=/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/skypilot-config.yaml
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/python \
  tpu/science/results/fresh-grpo-campaign-20260919/submit.py --hardware v4 --stage math
/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/python \
  tpu/science/results/fresh-grpo-campaign-20260919/submit.py --hardware v4 --stage science
```

Use `--hardware v5p` to choose the alternative pool. Use `--stage rglru` to
submit only its three models on the selected hardware. RG-LRU is not included
in `--stage science`; that stage remains qubit and circuit. The submitter checks
identities, fresh output paths and package hashes, uploads and reads back each
archive, and saves an attempt receipt before dispatch. It calls:

```
sky jobs launch -p POOL TASK_YAML --priority PRIORITY -y -d
```

Confirmed submissions are skipped on rerun; uncertain attempts stop for
reconciliation rather than producing duplicate jobs. Pools:
`tpuswarm-v4-64-central2-qwen35-erdos` and `tpuswarm-v5p32-east5a-erdos`.
Every task audits free storage, TPU ownership and leftover workload processes
before runtime startup. Runtime ownership uses systemd with checkpoint recovery.
Check job ID, assigned worker, bootstrap journal and optimizer checkpoints;
SkyPilot RUNNING alone does not establish training progress.

## Validation and deployment evidence

See [validation.json](validation.json) and [submissions.json](submissions.json)
for this deployment's test and receipt summaries. Local tests verify both
hardware contracts, bounded bootstrap recovery/admission, full-suite feedback,
CPU role allocation and legacy compatibility. The new bounded 32x16 path and
its v5p extension are not claimed hardware-proven before these jobs execute.
The six RG-LRU additions passed local config, initial-prompt, package and
topology tests, including trainer rows with a nonzero leader. Their owned-grader
bootstrap/transition has not yet been executed on TPU hardware.
Existing jobs/checkpoints are not inputs to this campaign and are not deleted
by either script.

## Submitted jobs

Snapshot at 2026-09-19T21:19:22.810676+00:00.

| Task | Qwen | Gemma | Muse | Priority |
|---|---:|---:|---:|---:|
| ac2 | 1227 | 1228 | 1229 | 100 |
| cp26 | 1230 | 1231 | 1232 | 100 |
| qubit | 1233 | 1234 | 1235 | 50 |
| circuit | 1236 | 1237 | 1238 | 50 |

All twelve were accepted by the v4-64 pool. Startup/queue state is recorded in submissions.json; no bootstrap completion or optimizer step is claimed yet. No v5p duplicate was submitted.
