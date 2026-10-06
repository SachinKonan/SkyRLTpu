# Pooled multi-LoRA on the GPT-OSS v6e executor

Integration branch: `agent/gptoss-multi-lora`, based on
`agent/tunix-multihost-gptoss` at `0f341111`. The original multi-LoRA work was
uncommitted in `SkyRLTpu-multi-lora` (base `86db6ec4`); this port carries its
feature changes onto the current executor rather than replacing the executor.
The Discover client has a companion `agent/gptoss-multi-lora-client` branch.

## What is implemented

For k same-base adapters, generate G/k rollouts per adapter and pool G total
rollouts per problem. Compute advantages over that pool. Every learner trains
on all G trajectories using the original generating adapter's token logprobs:
`ratio = exp(current_learner_logp - behavior_logp)`. CISPO uses a detached ratio
capped at `importance_cap` (default 2), including for off-policy trajectories.
This is a biased token-level clipped surrogate, not an exact trajectory
importance correction. Prompt and forced tokens have zero advantage.

`POST /api/v1/multi_lora_training` stores one `MultiLoraTrainingRequest` with
shared datums and source model/version metadata. A queue barrier drains earlier
operations and prevents later operations from crossing the bundle. The backend
executes forward/backward sequentially by default. With
`trainer.stacked_lora_training=true`, NNX vmap batches the independent adapter
losses and gradients over one shared frozen base.
Adapters retain separate weights, Adam states, and gradient accumulators. The
client drains all results before stepping any adapter. A failed bundle clears
all target accumulators and blocks training until restore/recreation; optimizer
updates are not a multi-model transaction.

The shared request reduces duplicated queue payloads. It still performs k
learner passes over G trajectories. Adapter weights, optimizer states and
accumulated gradients grow with k; the base is shared. Stacked execution also
increases activation workspace with k. The v6e profile keeps the base executor's 20,480-token microbatch
budget (2 x 10,240), not the earlier Qwen v4 budget of 2 x 22,528.

## Stacked Qwen validation on Asia v6e-32

`profiles/qwen35_v6e_32_stacked_lora_asia.json` enables k=2 vectorized
forward/backward, with independent optimizer objects and per-adapter clipping,
normalization, repeated-K/V corrections and Adam moments. Two rows per adapter
means four active 10,240-token rows. The existing FSDP=2 layout still partitions
the row dimension within each adapter; the adapter dimension is replicated.
It does not yet repartition FSDP over the adapter dimension.

The profile enables `stacked_lora_verify`: each real pooled batch is also replayed
sequentially without any optimizer update. It requires gradient relative L2
error <= 0.02 and maximum target-logprob absolute error <= 0.05, and restores
the stacked gradients for the actual independent updates. Failures poison the
whole cohort through the existing bundle failure handler. Trainer logs record
paired durations and errors. Verification doubles the training work, so use
the separate `stacked_seconds` and `sequential_replay_seconds` measurements;
whole-request throughput includes validation. First-call timings include JIT.

Both Qwen profiles use the checksum-verified v9 source bundle, existing Asia HF
and Orbax caches, and seed inference compilation from
`vllm-xla-cache-v6e-qwen35-tp4-s22528-v1`. Newly compiled inference programs are
shared via `vllm-xla-cache-v6e-qwen35-tp4-s16384-seq8-mem80-chunk8192-lora2-multi-lora-v1`.
The new stacked trainer graph has its own reusable compile prefix. JAX cache
keys determine compatibility; seeding does not guarantee all graphs hit.

CPU tests execute the production NNX/FLCE backward and complete microbatch
driver on a tiny shared-base model, including padding, accumulation, distinct
adapter outputs and optimizer updates with different hyperparameters. These
tests do not establish Qwen TPU kernel support or HBM fit; the runtime trial
must establish those separately.

Sampling clients switch to the next population only after all checkpoints are
published. `multi_lora_cohort.json` records the last complete population for
resume. Pooled trajectory archives retain source IDs, immutable sampling
versions, sampled tokens, masks, rewards, advantages and behavior logprobs.
`client/multi_lora_timings.jsonl` records sampling/grading, backward, optimizer
and publication timings, including per-target bundle timings.

## GPT-OSS adaptation

- Preserve the GPT-OSS abstract Qwix installation and sparse expert LoRA path;
  create additional adapters from the cached factor shapes without another base
  model. The opt-in `independent_lora_init` honors each adapter seed even after
  the original base state is released. Expert down-projection A uses the input
  width as fan-in in `[expert, layer, input, rank]`; it is not a dense A layout.
  Initialization preserves output sharding and starts every B at zero.
- Preserve attention/router PEFT export, expert sidecars, physical-slot RPCs,
  and the pinned TPU-inference fork. An upload replaces only its explicitly
  named predecessor when multiple adapters are enabled. Loading B cannot evict
  A through the old global last-upload marker.
- Raise the private profile's streamed upload limit to 8 GiB: GPT-OSS 120B
  expert adapters exceed the ordinary 2 GiB ingress limit.
- Ray Serve tracks a set of committed adapter versions, pauses admission during
  replacement, and restores the committed set before registering a replacement
  engine. Both paired engines retain every active adapter. The executor's
  TP4/PP2 placement groups and round-robin routing between engine pairs remain.
- Return full learner logprobs for own/cross importance-ratio diagnostics;
  pooled mode rejects the ordinary executor's minimal-backward-output setting.
- Use the preset's `gpt_oss_high_reasoning` renderer and sampling limits for
  every member. Pooled mode requires ingress routing and cannot be combined
  with the separate carried/fresh learnable LoRA mix.
- Preserve the Qwen repeated-K/V-head correction, scoped to Qwen models;
  GPT-OSS does not enter that path.

Fresh adapters initially implement the same base policy because B is zero.
Independent A seeds allow their subsequent updates to diverge; useful sampling
diversity is not guaranteed merely by choosing different seeds.

## Reviewable trial

`profiles/gptoss120b_v6e_32_multi_lora.json` inherits the GPT-OSS v8 base bundle
and v6e layout: eight hosts, four TP8/FSDP2 trainer hosts selected by topology,
and four inference hosts forming two TP4/PP2 engines. It sets k=2, G=32, two
iterations, one problem group, rank 32, two serving LoRA slots, memory
utilization 0.8, and a private runtime root and compile-cache destinations.
Memory fit at 0.8 for this population has not been measured.

Build locally (no upload or launch):

```bash
python -m tpu.swarm.ray_train.build \
  tpu/swarm/ray_train/profiles/gptoss120b_v6e_32_multi_lora.json \
  --output /path/to/build
```

The build includes an explicit file manifest with SHA256 hashes for backend,
API, client and upload-handler changes. Worker source paths, trainer/serving
environments and the locked client environment include the overlay identity,
so an old cached installation cannot silently run the pre-port client.
Ordinary single-adapter builds retain the base-bundle path.

## Evidence and remaining hardware gate

Prior Qwen evidence: v4-32 job 369 passed the corrected short-context
four-adapter serving audit; job 373 passed cached-trajectory trainer replay on
all four ranks (115 checks, eight optimizer updates, frozen base and checkpoint
round trip). Those were separate serving and training tests. The live k=2
Qwen v4-64 trials failed during inference before a bundled optimizer step.

The GPT-OSS branch records a separate passed 20B concurrent multi-LoRA gate on
v6e (Jobman 000705), including expert-slot reuse and base parity, in
`docs/gpt-oss-mxfp4-lora.md`. This does not establish 120B TP4/PP2 behavior.

Before calling the new path hardware-validated, run representative 120B mixed
adapter requests on both pairs, complete a bundled backward and optimizer step
per adapter, publish/reload both adapters, and measure HBM plus first/warm
iteration timings. This integration work does not launch or modify fleet jobs.

## CPU validation of this port (2026-09-09)

The focused suite passed 111 tests with one opt-in installation smoke skipped.
It covers the real API/SQLite queue barrier, shared HTTP submission/retrieval,
pooled training phases and checkpoint publication, actual Tunix create/step
methods on tiny NNX states (including expert-shaped leaves), eight-device CPU
sharding, repeated-K/V canonical gradients, real FastAPI uploads and Ray Serve
admission, paired routing, cache identity, executor commands/topology contracts,
and the existing GPT-OSS serving-gate harness tests.

After adding the full-backward-output guard, all 34 profile/command/serving
checks passed. The additional guard brings the distinct passing test count to
112. Raw JUnit reports are in the local `gptoss-multi-lora-build-001` artifact
directory. The optional skipped test installs the entire locked client from a
predownloaded uv cache.

This was not a full repository test run: the pre-existing lifecycle test imports
removed `OLD`/`NEW` symbols from `patch_maxtext.py`, and the CPU environment lacks
Tunix for the full imported-backend LoRA-mix suite. The targeted backend tests
execute the production methods via AST extraction; they do not instantiate the
120B model. Hardware validation and throughput remain outstanding.
