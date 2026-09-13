# RecurrentGemma kernel generation

`TTD_ENV=recurrent_gemma` selects RG-LRU in `tpu/run_ttd_ensemble.py` and the
Ray v2 training executor. `recurrentgemma`, `pallas_rglru`, and `rg_lru` are aliases.
The generator model can be Qwen3.5 or another supported model; RecurrentGemma
names the **kernel task**, not the model being trained.

The task optimizes `kernel(x, a, reset) -> h` with gates already computed:
`a_eff = a * (1-reset)` and `h_t = a_eff_t*h_(t-1) + sqrt(1-a_eff_t²)*x_t`.
Both outputs and gradients with respect to `x` and `a` must be correct.
Generated programs must contain a real Pallas kernel. They are sent as source
to the Arena queue and never executed on the training or inference hosts.

The initial state contains the linked branch's time-blocked affine-scan seed.
It has **no asserted speedup**. Its raw score starts at zero until a generated
candidate is graded. Successful candidates enter Discover's existing state
pool with their source, measured combined speedup, and compact judge feedback.
GRPO/TTD and multi-adapter sampling use that same environment interface.

## Grading and reward

The default suite is the six single-chip probe shapes declared by the judge,
including `probe-holdout-2x1500x2560`. The prompt reads the reference function
and literal shape declarations from `judge/problems/rg_lru.py` without importing
JAX on the training client. The large production shapes and TP4/TP8 cases
remain in the benchmark but are outside this initial RL suite.

Run the ported **per-case Ray judge pool** with `--baseline all`. Each shape
gets a fresh TPU process. Forward and backward correctness must pass. The
environment uses `reward_with_bwd` as its training reward and the geometric
mean of the measured forward/backward ratios as its state value. This keeps
the raw speedup separate from the Arena noise-floor gating. The denominator
is the fastest available calibrated honest implementation at each shape,
including the production RecurrentGemma scan and XLA associative scan.

A candidate failure receives zero. A missing, mismatched, or incomplete
verdict raises an infrastructure error instead of becoming a training reward.
In particular, a passing aggregate with excluded shapes or missing backward
timings is rejected. The environment requires `baseline_mode=all`; older pool
versions without this metadata cannot silently substitute their scoring rules.
Complete returned verdicts, submitted source, parent IDs, and work IDs are saved
under `<client log_path>/arena/<request tag>.json` (inside the run's client tree,
which the executor already snapshots).

## Configure a run

The example profile is
`tpu/swarm/ray_train/profiles/qwen35_recurrentgemma_v5p_32.json`:

- Qwen3.5-27B, one adapter, GRPO mean-baseline advantages.
- One group of 8 rollouts per step, two steps, 3600-second grading wait.
- Existing Ray v2 trainer and vLLM Ray Serve inference.
- A separate run root and compile-cache write prefixes. HF/Orbax model caches
  are reused; the inference cache is seeded from the existing compatible cache.
  The trainer uses a fresh compile prefix.

Replace `client_env.ARENA_QUEUE_URL` (`http://arena-queue:8791` is a placeholder
hostname) with a queue reachable from the training head, and use a new run ID.
The queue and TPU judges must already be running on dedicated capacity.
Building this profile does not start them, allocate TPUs, or submit training:

```bash
python -m tpu.swarm.ray_train.build \
  tpu/swarm/ray_train/profiles/qwen35_recurrentgemma_v5p_32.json \
  --output /tmp/recurrentgemma-build
```

The bundle includes a hash-checked Arena client source overlay even for one
adapter. With multiple adapters it combines the Arena and multi-LoRA overlays;
their digest is part of the worker source identity, preventing reuse of an old
source tree. The new task also works with another validated accelerator profile:
copy the `client_env` settings while retaining that profile's topology/cache
settings. Do not point it at an existing job's run ID or mutable run directory.

For a directly launched ensemble client, export `TTD_ENV=recurrent_gemma`,
`TTD_PROBLEM_TYPE=rg_lru`, `ARENA_QUEUE_URL`, and `EVAL_TIMEOUT=3600` alongside
the usual model/trainer settings. The evaluator bypasses the generic CPU code
runner regardless of `TTD_EVAL_BACKEND`.

## Dedicated judge setup

Use this worktree's Arena code on a judge host with the existing Arena TPU
environment (JAX/libtpu, Pallas, RecurrentGemma, Ray, FastAPI/uvicorn). See
`../phase2/provision_judge.sh` for dependency context; do not run that provisioner
on a training host. The following commands are examples, not launch actions:

```bash
# Queue process on a reachable CPU host. Keep the service within your cluster network.
PYTHONPATH=tpu python -m pallas_arena.judge.queue \
  --host 0.0.0.0 --port 8791 --lease-timeout 1800

# In the judge environment, with a dedicated Ray cluster exposing its TPU resources:
export PYTHONPATH=tpu
cases=$(python -c 'from pallas_arena.rl.task import public_contract; print(",".join(n for n, _ in public_contract()[1]))')
python -m pallas_arena.judge.ray_pool \
  --queue http://QUEUE_HOST:8791 --problems rg_lru \
  --cases "rg_lru=$cases" --chips 4 --max-tp-width 1 \
  --ray-address auto --baseline all \
  --cache /path/to/arena-recurrentgemma/rewards \
  --compile-cache-dir /path/to/arena-recurrentgemma/jax
```

`--chips` must match the judge host; no chips are taken from the training
executor automatically. Reward caches belong to the judge and include its
device/problem/contract identity. JAX compilation caches also live on the
judge, separate from the trainer and vLLM caches. Supply `--jax-cache-gcs` only
when you have chosen a compatible judge cache prefix. The queue itself is
in-memory; the client resubmits lost work after a restart.

## Provenance and validation

### Bounded sampling on one v5p-32

`qwen35_rglru_sampling_v5p_32.json` runs through the Ray v2 executor with no
trainer. Head rank 0 hosts the localhost Arena queue and an isolated Ray
judge runtime owning its four chips; workload Ray advertises zero head TPUs.
Ranks 1-3 each serve Qwen3.5-27B TP4 through the existing Ray Serve ingress.
The executor owns service startup, monitoring, artifact writeback, and cleanup.

The first trial requests 24 independent samples (12 concurrent requests,
temperature 0.8, top-p 0.95, 12288 output-token cap) plus the unchanged seed.
Prompts include the earlier v4 backward-reversal failure. Every submitted
program targets all six probe cases, including the holdout, with mandatory
backward correctness and 20 timing pairs against the fastest honest baseline.
Rejected candidates are results; missing verdicts or judge faults are failures.
No optimizer steps or adaptive search are performed in this sampling trial.

Sources, full responses, verdicts, token counts, durations and `summary.json`
are under `<run>/client/arena`, mirrored to the run's GCS prefix. The judge
uses a private JAX 0.10.2 / RecurrentGemma 1.0.1 environment and local
`<root>/arena-cache/{jax-0.10.2,rewards}` caches. The inference hosts reuse the
Qwen HF cache and seed a new compile-cache prefix from the existing compatible
vLLM cache. The first run is SkyPilot job 585 on pool worker 367; submission
alone does not establish readiness or candidate correctness.

Based on `agent/gptoss-multi-lora` at `9665d86a`. The Arena judge updates,
per-case collector/pool, and seed were ported from
`origin/agent/ttd-discover-erdos` at `56a5cb73`. Its referenced Discover submodule
commit `9b254228` was unavailable from the configured remote, so this integration
uses the current Discover interface and leaves its source unchanged.

RG-LRU's problem version is bumped to 3 for mandatory gradient correctness,
separating verdict caches from the older soft backward penalty contract.
This integration does not claim a TPU speedup or a completed RL run. CPU tests
cover the queue-to-reward path, infrastructure failures, source packaging, and
the seed's forward/backward semantics under Pallas interpretation. Real Mosaic
compilation, timing, and training throughput require a dedicated TPU smoke run.
