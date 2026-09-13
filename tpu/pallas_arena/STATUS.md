# Pallas Arena integration status

RecurrentGemma's RG-LRU is now selectable as `TTD_ENV=recurrent_gemma` in the
Ray v2 training client. See [configuration, grading, and cache details](rl/README.md).

This worktree ports the linked branch's per-shape Ray judge and combined
forward/backward reward into the current multi-LoRA base. Gradient correctness
is a hard gate for RG-LRU; its problem/cache version is 3. Initial RL coverage
uses six single-chip probe shapes, including ragged time, and retains all
existing production/TP benchmark cases for later validation.

The example Qwen3.5 profile builds an immutable source overlay. Its queue
hostname must be replaced before launch. No TPU jobs or training experiments
were submitted with the initial code integration. Historical speedups in the
upstream branch's STATUS.md are not validation of this worktree.

After the user allocated one east v6e-32 worker, SkyPilot job **506**
(`recurrentgemma-v6e-smoke-001`) was submitted to
`tpuswarm-v6e32-east5b-qwen35`. It grades the seed at time tiles 512 and 256
over the six probe shapes. This is a kernel correctness/timing smoke test,
not an RL training run. Artifacts are under
`gs://sk7524-tinker-tpu-us-east5/arena-runs/recurrentgemma-v6e-smoke-001`.

Local validation: the 97-test integration/runner batch passed. The broader
105-test judge/reference batch initially had 9 export failures caused by a
missing `flatbuffers` dependency in the local test environment. With that
dependency supplied in an isolated directory, all 31 tests in the export
worker + new environment batch passed. TPU results are pending. Job 506 was cancelled before execution after its
assigned worker 327 disappeared from both GCP and the pool. Replacement job **509**,
`recurrentgemma-v6e-smoke-002`, was submitted only after cancellation completed;
its artifact prefix ends in `recurrentgemma-v6e-smoke-002`. At most one worker
is allocated to this experiment.

Job 509 is assigned to worker 322. GCP reports its VM READY/HEALTHY. The
SkyPilot API execution request `b507bed4-e76a-4a97-b796-d8fa0dd5ae3e` is PENDING
in dispatch; the smoke script has not started yet. This is a controller wait,
not a kernel failure. No training/inference jobs were changed.

On 2026-09-09 the user redirected the smoke test to the spare v4-32. Job
**564** (`recurrentgemma-v4-smoke-001`) is assigned to worker **103** in
`tpuswarm-v4-32-central2-smoke`, `us-central2-b`. GCP was READY/HEALTHY and
all four hosts passed an empty-device ownership audit. The task matches the
existing TPU-VM runtime, queued-resource flag, and 300-GB resource specification;
its source bundle matches all 85 Python files in this worktree.

As of 20:48 UTC, its SkyPilot execution request
`4d5fa468-2a61-4272-a4e7-e7de75281aed` is PENDING with no execution PID.
No v4 kernel result is available yet. Launch YAML and metadata are in
`/scratch/gpfs/ZHUANGL/sk7524/.cache/recurrentgemma-v4-smoke-001/`;
results will be written to
`gs://sk7524-tinker-tpu-us-east5/arena-runs/recurrentgemma-v4-smoke-001`.

The v4 run 564 subsequently failed before kernel execution because the private
smoke environment lacked `jaxtyping`. Retry **573**
(`recurrentgemma-v4-smoke-002`) installed `recurrentgemma[jax]==1.0.1`
with JAX 0.10.2 pinned and validated imports, reusing the identical source
bundle and worker 103. After about seven minutes waiting for API dispatch,
the smoke ran on all four hosts with actual TPU device ownership.

Run 573 completed its harness and uploaded all **12** unique shape/tile
verdicts (six shapes, tiles 256 and 512). SkyPilot reports SUCCEEDED, but
**kernel validation failed 12/12 at `aot_export`**. All report
`NotImplementedError: Unimplemented primitive in Pallas TPU lowering for tc: rev`.
The seed backward kernel uses `jnp.flip` at `rl/seed_rglru.py:109-112`.
Reference baselines compiled and ran; the candidate did not reach correctness
or performance validation. No seed code fix has been applied.

Artifacts and a compact `summary.json` are saved under
`/scratch/gpfs/ZHUANGL/sk7524/.cache/recurrentgemma-v4-smoke-002/`;
remote verdicts and logs are at
`gs://sk7524-tinker-tpu-us-east5/arena-runs/recurrentgemma-v4-smoke-002/`.

User authorized exactly one v5p-32 for sampling/grading through SkyPilot.
Job **585**, `qwen35-rglru-sampling-v5p-32-001`, owns pool worker **367**
in `tpuswarm-v5p32-east5a-erdos`, us-east5-a. All four hosts' VFIO devices
were unowned before submission; provider node READY/HEALTHY and queued
resource ACTIVE. Immutable executor bundle SHA256:
`f94db20354f71a1be789d087806895e9ff762854d86dce92771a6d6b6780571e`.
Head rank 0 runs the queue and private local Ray judge; ranks 1-3 run
Qwen3.5-27B TP4 behind Ray Serve. There is no trainer. The automatic client
will generate 24 samples and grade the unchanged seed after engine readiness.
Focused tests passed 21/21. The preexisting lifecycle test could not collect
because it imports absent `patch_maxtext.OLD`; it was not counted as passing.

While inference compiled, an extra unchanged-seed control was submitted to
this same judge: `w000000-c40f82e9`, tag `seed-preflight`. It ran the production
reference on v5p and failed candidate AOT export with the same unsupported
`rev` primitive as v4. The pool returned the rejection in 36.9 seconds and
cancelled the five sibling shapes after the fatal compile error. This is not
six completed shape tests or a candidate performance measurement. Result is
saved in `client/arena/seed-preflight.json` locally and in the GCS run prefix.
As of the last observation, all three engines were compiling their backbone
shapes with no restarts; generated-candidate results remain pending.
Artifacts: `/scratch/gpfs/ZHUANGL/sk7524/.cache/recurrentgemma-v5p-sampling-001`.
GCS: `gs://sk7524-tinker-tpu-us-east5/ray-training/qwen35-rglru-sampling-v5p-32-001`.

The seed's backward scan now composes affine maps as a direct suffix scan,
using static shifts toward later timesteps. It no longer emits `rev` inside
Pallas, and retains the existing reverse block traversal and carry semantics.
An isolated CPU environment with **JAX/jaxlib 0.10.2** reproduced the old
seed's `aot_export` failure and exported the fixed seed successfully for all
six shapes at tile sizes 256 and 512: **24 forward/backward TPU-target
artifacts**. Forward and both-gradient numerical tests also passed on that
pin (ragged time, resets, and near-unit gates). The export regression test is
now in `tests/test_recurrent_gemma_env.py`. CPU export is not TPU execution
or a performance measurement.

Hardware test **592**, `recurrentgemma-rev-fix-v5p-001`, requests one v5p-32
in the existing pool. Worker 389 was audited empty and READY/HEALTHY, but
recovering job 577 acquired it before this test could. Job 592 remains
PENDING without a launched workload at the latest check. Its immutable
Arena source bundle is
`c2fe8f6e5e1e3ea977e864f7e07b6420b0421fcea31eae235432037080aff3e1`.
Artifacts and exact-version export results are in
`/scratch/gpfs/ZHUANGL/sk7524/.cache/recurrentgemma-rev-fix-v5p-001/`.

Job 585 subsequently finished FAILED: all three engines reached readiness,
but every generation request received HTTP 400 because the sampling client
sent per-request `seed`, which this TPU ingress explicitly rejects. No model
samples were generated. The sampler now omits that field, keeps stochastic
temperature/top-p sampling, and saves server error bodies. Its private judge
CPU reservation now permits four case tasks rather than serializing them
behind a 32-CPU limit (each requested 26 CPUs on the v5p host). These sampler
fixes are local and have not been validated in another inference run.


## 2026-09-11: v4 three-model sampling comparison

Job 592 is now SUCCEEDED on v5p worker 331. Its hardware verdicts show the
fixed tile-256 seed passing all six shapes (forward and both gradients).
Tile 512 fails backward VMEM: 19.13 MiB requested against a 16 MiB limit.
The default seed time tile is now 256. These results supersede the pending
snapshot above; v4 execution of the corrected seed still needs validation.

`rglru_three_models_v4_32.json` runs one head-local RG-LRU judge and three
Ray v2 / Ray Serve TP4 engines: Qwen3.5-27B, Gemma4-31B, Muse-Glimmer-30B.
Each gets 32 samples, concurrency 8, temperature 0.8, top-p 0.95, and 8192
output tokens including reasoning. All receive the same task and seed, with
model-specific prompt framing, explicit BOS where required, and stop tokens.
The corrected seed must pass the full target-hardware grading contract before
sampling. Eight 64-token warmup requests per model are recorded separately.
Sampling starts after all engines finish warmup; initialization and warmup are
excluded from the generation timing. Additional shape compilations during the
measured workload are included. This is a bounded generation/grade comparison,
with no training or feedback into subsequent candidates.

Host caches use gcloud storage and tmpfs (128 GiB cap, 128 GiB runtime reserve).
Existing central2 weight caches and per-model v4 compilation seeds are restored.
New compilation entries write to experiment-specific prefixes. Judge compilation
is keyed by accelerator and JAX 0.10.2 under the experiment root. Reward caching
is disabled so each model's best valid candidate can be retimed independently.
Artifacts are under `client/arena/` in the GCS run prefix; infrastructure errors,
invalid kernels, and truncated answers remain explicit in the per-model report.
The worker root is `~/.cache/skyrl-ray-rglru-v4-compare`.

Submitted as SkyPilot job **633**, assigned to audited-empty v4 worker **104**.
Source SHA256: `9019ca9fddae7550d5586de5832e30dab7557f4c76f6a4a2abc15283fb949b32`.
Run artifacts: `gs://sk7524-tinker-tpu-us-central2/ray-training/rglru-three-models-v4-32-001/client/arena`.
At submission, no new inference samples or v4 kernel verdicts are available.

Job 633 failed during cache preparation, before any engine or sampler started.
Muse uses materialized snapshot files; this worktree lacked the existing v4
integration loader support for that layout. Ported that cache-only fix and its
9 HF mirror tests. All 48 cache/sampler checks passed. Read-only GCS preflight
verified Qwen's 24 manifest files, Gemma's 9, and Muse's 13 snapshot files
including both weight shards (59,581,829,216 bytes total). Retry uses run ID
`rglru-three-models-v4-32-002` and source SHA
`47161273ef3c8d54aecb2d0b8fdcbcaccf1ef5a6e73dc58833ea9198eb74a6ba`.

Job 634 restored all model weights and compilation seeds and passed every
serving import check, then failed because `uv venv` refused the incomplete
judge environment left by job 633. No inference samples ran. Judge setup now
recreates its owned incomplete environment with `uv venv --clear`; controller
startup observes judge and cache-preparation failures together before deployment.
Twenty relevant tests passed. Retry run: `rglru-three-models-v4-32-003`.

## Qwen v4 convolution fix

Job 637 (run 003) exhausted Qwen's engine restart budget on the unsupported
Pallas causal-convolution bf16-to-f32 interleaved unpack. Muse reached serving
readiness; Gemma was compiling; the comparison produced zero samples.
The v4 integration commit `a0022681` already handles this exact failure.
The comparison profile now enables `ragged_conv1d` for the Qwen host, exporting
`USE_JAX_RAGGED_CONV1D=1`, and seeds compilation from the existing v4 JAX-conv
cache. New writes use a separate JAX-conv prefix. The 4096 prefill cap and
0.8 memory utilization remain the comparison settings. A regression checks
the emitted environment and keeps the v5p Qwen default unchanged.
Prepared run ID: `rglru-three-models-v4-32-004`; not submitted with this fix.

## Muse native registration fix

Job 656 reached all three serving-ready checks and the seed passed all six
v4 cases, including both gradients (combined raw speedup 1.1572210863x).
The job then failed on Muse's warmup, before the 96 measured samples. The
restrictive VLLM_PLUGINS list excluded register_layers, so Muse resolved to
TransformersMultiModalForCausalLM and hit NonConcreteBooleanIndexError in
`llm_input_ids[multimodal_mask] = 0`.

Ported the v4 integration `unset_plugins` configuration and environment fix:
Muse now removes VLLM_PLUGINS after parent/profile values are applied. All
three inference preset dictionaries match the v4 integration checkout; the
comparison retains its explicit sampling/memory overrides. Fifty-five focused
tests passed, including stale plugin restrictions and per-host configuration.
A CPU-only check inside the actual worker serving environment, with the list
unset, registered and imported the native class
`tpu_inference.models.vllm.muse_glimmer.MuseGlimmerForCausalLM`. This verifies
registration/import, not TPU generation. Prepared run 005 uses a fresh native
Muse writeback prefix and retains the existing native compilation seed.
The corrected comparison has not been submitted in this fix turn.
