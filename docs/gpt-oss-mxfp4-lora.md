# GPT-OSS LoRA on TPU: immutable MXFP4 experts

Status: training and inference implementations are complete, including
per-request multi-LoRA routing through the GMM path; CPU numeric/integration
tests are included. The 20B trainer acceptance gate passes on a four-host
v6e-16 with TP=8/FSDP=2.
The 20B vLLM load/inference/concurrent-multi-LoRA gate also passes on one
v6e-8 with TP=8.
Colocated trainer-to-vLLM weight-sync validation and the 120B v6e-32 gate
remain before a long RL run.

## What is actually quantized

GPT-OSS does **not** have an MXFP4 router. OpenAI quantizes the MoE expert
projection weights to MXFP4 (4.25 bits/parameter); the router is a standard
linear projection. The published checkpoint explicitly excludes
`model.layers.*.mlp.router`, attention, embeddings, and `lm_head` from MXFP4.

- [OpenAI GPT-OSS architecture and quantization](https://deploymentsafety.openai.com/gpt-oss/architecture)
- [GPT-OSS 20B checkpoint config](https://huggingface.co/openai/gpt-oss-20b/blob/main/config.json)

The 20B model has 24 layers, 32 experts, top-4 routing, hidden size 2,880,
and expert intermediate size 2,880. The 120B model has 36 layers and 128
experts, with the same hidden size and top-4 routing.

## The old path and why it is removed

The former implementation loaded a dequantized BF16 checkpoint, formed each
dense `A @ B` expert delta on the host, and incrementally mutated the fused
base tensors:

```text
base <- base + delta(new) - delta(previous)
```

That path defeated the memory purpose of the native checkpoint, made adapter
replacement stateful, accumulated BF16 rounding error, and coupled the upload
protocol to the current contents of the base buffers. The old worker RPC and
its merge implementation have been deleted.

The new invariant is:

```text
MXFP4 base expert tensors never change.
An adapter upload only replaces fixed-shape BF16 factor buffers.
```

## Numerical contract

Tokens are already sorted by selected expert before the TPU GMMs run. The
adapter reuses those exact `group_sizes` and offsets.

For each expert and routed token:

```text
z_gate = MXFP4_GMM(x, W_gate) + scale * (x A_gate) B_gate[e]
z_up   = MXFP4_GMM(x, W_up)   + scale * (x A_up)   B_up[e]
h      = GPT_OSS_SWIGLU(z_gate, z_up)

y_base = MXFP4_GMM(h, W_down)
y_lora = scale * (GMM(h, A_down[e])) B_down
y      = topk_weighted_reduce(y_base + y_lora)
```

The activation must not be fused into the base W13 GMM when expert LoRA is
enabled. The implementation requests `fuse_act=None`, adds both BF16 deltas,
then invokes the same `swigluoai` helper used by the base kernel. W2 LoRA is
added before unpermutation and top-k weighted reduction. This matches the
ordering used by vLLM's GPT-OSS MXFP4 expert implementation on GPU.

The MaxText/Qwix factors are asymmetric because one side is shared across
experts:

| Component | Exported A | Exported B | TPU buffer layout |
| --- | --- | --- | --- |
| gate (`wi_0`) | `(H, R)` | `(R, E, I)` | A `(S,Hpad,Rmax)`, B `(S,E,Rmax,Ipad)` |
| up (`wi_1`) | `(H, R)` | `(R, E, I)` | A `(S,Hpad,Rmax)`, B `(S,E,Rmax,Ipad)` |
| down (`wo`) | `(E, I, R)` | `(R, H)` | A `(S,E,Ipad,Rmax)`, B `(S,Rmax,Hpad)` |
| router | `(H, R)` | `(R, E)` | ordinary PEFT `mlp.router` tensors |

`S` is vLLM's `max_loras` physical-slot count. Rank is padded to
`max_lora_rank`, so uploads replace values without changing the JAX pytree or
causing slot- or rank-specific compilation.

## Parallelism

The implementation supports the two GMM backends used for TPU inference:

- GMM TP: gate/up B are sharded on intermediate output; down A is sharded on
  its intermediate input. Shared A/B factors are replicated. Each TP shard's
  down-LoRA result is a linear partial and is included in the existing final
  `psum`.
- GMM EP: every factor with an expert dimension is expert-sharded; shared
  factors are replicated. The same local expert offset used by the base GMM is
  used by LoRA GMMs.

The custom fused EP kernel is rejected when expert LoRA is present; it does not
yet expose the required pre-activation and pre-reduction insertion points.

## Adapter lifecycle

1. Model loading allocates zeroed BF16 factor buffers on their final TPU
   shardings. MXFP4 blocks/scales remain compressed in the source checkpoint.
   TPU v7+ may execute native FP4 GMMs; v6e dequantizes once during loading and
   requantizes the expert weights to the supported FP8 GMM runtime format.
2. The trainer exports attention and router weights in
   `adapter_model.safetensors`. Expert factors remain in
   `moe_lora.safetensors` with `gptoss-moe-lora/v1` metadata.
3. vLLM loads the ordinary PEFT adapter and assigns it a physical Punica slot.
   The upload endpoint then calls `set_moe_lora_factors` on every worker with
   that adapter ID. The RPC
   reads the worker-local safetensors file, validates shapes/ranks, transposes
   the serialized `(R,E,I)` B tensors to the GMM's `(E,R,I)` layout, pads them,
   and replaces only the matching factor-bank slot.
4. Request-time Punica metadata supplies one physical slot per token. After
   top-k routing, rows are grouped by `(expert, slot)` for the skinny expert
   GMMs; slot `-1` is the base model and contributes exactly zero delta.
5. vLLM's LRU activation callback keeps expert-bank ownership synchronized
   when an adapter moves to a different physical slot. Adapters without expert
   factors zero only their assigned slot.

The original v1 live gate below proved safe single-adapter replacement. The
multi-LoRA v2 gate additionally keeps A and B resident, mixes simultaneous
base/A/B requests, classifies them against homogeneous concurrent reference
families, and forces LRU slot reuse to catch stale expert factors.

## Training reality

The SkyRL MaxText fork adds the expert path that generic Qwix cannot create:
Qwix still wraps attention and the BF16 router, while
`install_sparse_expert_lora()` attaches six `nnx.LoRAParam` factors to every
scanned `RoutedMoE`. MaxText keeps `sparse_matmul=true, megablox=true`; the
factor branches consume the already-routed token buffers inside the same
`shard_map`. There is no dense-expert fallback and no `E / top_k` excess base
MoE work.

Training is pinned to
[`SachinKonan/maxtext@d388c54`](https://github.com/SachinKonan/maxtext/commit/d388c5478b18b2322ab36c032deb87b9a4ff065f)
(`skyrl/gptoss-sparse-lora`), based on upstream MaxText `f6dea15`.
This revision also exposes the sown final hidden state to Tunix when vocabulary
tiling suppresses the full logits tensor.

The stored factor axes preserve MaxText's layer scan metadata. On the proposed
v6e-32 mesh (`tensor=8`, `fsdp=4`), gate/up B and down A shard their intermediate
dimension over tensor parallelism; gate/up A and down B shard the model
dimension over FSDP outside the sparse `shard_map` and are gathered at its
existing boundary. The final down-LoRA partial uses MaxText's existing TP
reduce-scatter.

Rank-32 expert plus attention/router adapters for 120B are about 2.4 GiB in
BF16 globally before optimizer state. This is suitable for LoRA training on a
v6e-32 (1,024 GB raw HBM total); full-parameter Adam training is not.

## v6e-32 120B training recipe

Use the whole eight-host slice for the trainer during the initial compile and
forward/backward proof. Sampling should run on a separate warm slice until a
mixed trainer/inference partition is measured for 120B.

```bash
MODEL_NAME=openai/gpt-oss-120b
TUNIX_MAXTEXT_MODEL_NAME=gpt-oss-120b
TUNIX_MAXTEXT_PIP_SPEC="maxtext @ git+https://github.com/SachinKonan/maxtext.git@d388c5478b18b2322ab36c032deb87b9a4ff065f"

TRAIN_WORKERS=0,1,2,3,4,5,6,7
TP_SIZE=8
FSDP_SIZE=4
TUNIX_ROW_SHARD=4
TRAIN_TPU_PROCESS_BOUNDS=4,2,1
TRAIN_TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1

LORA_RANK=32
TUNIX_MAX_TARGET_LENGTH=1024
TUNIX_UNIFORM_SEQ_LEN=1024
TUNIX_TRAIN_TOKEN_BUDGET=4096
TRAIN_MICRO_BATCH_SIZE=1
TUNIX_FLCE_TILE_SIZE=512
TUNIX_MAXTEXT_KWARGS='{"sparse_matmul":true,"megablox":true,"num_vocab_tiling":64,"remat_policy":"full","allow_split_physical_axes":true}'
```

The runnable TPUSwarm profile is
[`tpu/swarm/examples/v6e32-gptoss120b-smoke-pool.yaml`](../tpu/swarm/examples/v6e32-gptoss120b-smoke-pool.yaml).
Its checkpoint is staged once by
[`tpu/swarm/stage_gptoss120b_orbax.sbatch`](../tpu/swarm/stage_gptoss120b_orbax.sbatch)
from the pinned 233.7 GB BF16 source. The 256 GiB CPU job uses the streaming
converter at `SachinKonan/maxtext@5e916f8`; live measurements exceeded 160 GiB
on the largest scanned expert leaf even after removing the redundant full-size
copy. A marker is written only after the Orbax upload completes. Every TPU host
requires that marker and restores the same checkpoint onto a 400 GB boot disk;
distributed startup never attempts eight independent HF conversions.

The first acceptance run should use four equal 1,024-token rows, which matches
FSDP=4 and avoids conflating sparse-LoRA correctness with long-context memory.
After zero-adapter parity and a nonzero-gradient update pass, increase sequence
length and token budget independently. Do not start at the production context
length: compile time and activation HBM, rather than model weights, become the
dominant uncertainty.

## Live 20B vLLM acceptance

### Concurrent multi-LoRA gate (v2)

The multi-LoRA gate passed on 2026-09-05 on Jobman job `000705`, TPU
`sk7524-v6e8-gptoss20b-vllm-multilora-e5b_1` in `us-east5-b`, using all eight
v6e chips as TP=8. The durable result is:

```text
gs://sk7524-tinker-tpu-us-east5/v6e-smoke-results/gptoss20b-vllm-mxfp4-multilora-v6e8-v2.json
generation: 1788642093978801
```

It reports `acceptance_pass: true`. Expert A and B were resident
simultaneously in physical slots 1 and 2. Across eight concurrent
base/zero/A/B batches, all 32 requests were closest to the correct homogeneous
reference family; the minimum separation from the next-wrong family was
`5.2864` against a required `0.25`. The gate also forced LRU slot reuse,
verified that the new non-expert occupant did not inherit stale expert
factors, exercised ordinary router LoRA, and restored exact base parity.

The accepted source/runtime pins are:

- SkyRL bundle source: `28c72db33a0ca84d60e3d852088b12d26ac2279f`.
- TPU inference fork: `b9e4024b5624fe74d7486c1b8dc34b1ce45c8aaa`.
- Immutable bundle:
  `gs://sk7524-tinker-tpu-us-east5/code-bundles/gptoss20b-vllm-mxfp4-multilora-v6e8-28c72db3.tar.gz`.
- Bundle SHA-256:
  `3d998eaf91b3f47aeb92e1ac76b08c7e67a67ed4ecfa22a9d171d33597f5092c`
  (GCS generation `1788641919779368`).

The first version of the v2 gate compared isolated and concurrent logprobs
bitwise. Live testing showed that this is not a valid invariant of the current
TPU runtime: even simultaneous identical base-model requests can produce
batch-position-dependent logprob differences. The accepted gate therefore
builds homogeneous concurrent reference clusters and verifies adapter-family
separation under the same batch shape. A real v6e microtest separately showed
exact mixed-slot W13 GMM parity and W2 parity within one BF16 quantum
(`1.5258789e-5`).

### Earlier single-adapter replacement gate (v1)

The vLLM gate passed on 2026-09-03 on Jobman job `000702`, TPU
`sk7524-v6e8-gptoss20b-vllm-lora-e5b_1` in `us-east5-b`, using all eight v6e
chips as TP=8. The canonical result is:

```text
gs://sk7524-tinker-tpu-us-east5/v6e-smoke-results/gptoss20b-vllm-mxfp4-lora-v6e8-v1.json
generation: 1788461724531989
```

It reports `acceptance_pass: true` and all seven semantic checks pass:
zero-adapter/base parity, nonzero expert effect, A-to-B replacement, exact
A replay, nonzero router effect, clear/base parity, and immutable base weights
after every swap. Each expert upload updated all 24 layers and reported
`base_weights_mutated: false`. The live sequence itself completed in 27.79
seconds after server startup and first-request compilation.

The accepted source/runtime pins are:

- SkyRL bundle source: `e5a753919327c7bbc45a1dfd948935d583aeac0f`.
- TPU inference fork: `22d9fcc6c23a536d1fb288b6aba02adbb24cb913`.
- Immutable bundle:
  `gs://sk7524-tinker-tpu-us-east5/code-bundles/gptoss20b-vllm-mxfp4-lora-v6e8-e5a75391.tar.gz`.
- Bundle SHA-256:
  `ac8b31d9c6962de47dd8aa187f4efbd65ddc984d6ee72ffa484df70a2ca17648`
  (GCS generation `1788461524492466`).

The live bring-up exposed two native-loader assumptions that unit tests had
not exercised on v6e:

1. GPT-OSS 20B's intermediate width 2,880 produced six W2 scale blocks at the
   former fixed block size 512, which cannot shard over TP=8. The loader now
   chooses a 384-value W2 block, yielding eight scale blocks and padding the
   intermediate dimension to 3,072.
2. Native `float4_e2m1fn` GMM kernels require TPU v7+, while v6e's Mosaic
   compiler rejects that vector type. The native MXFP4 loader now honors the
   standard `MOE_REQUANTIZE_WEIGHT_DTYPE` and
   `MOE_REQUANTIZE_BLOCK_SIZE` settings; the v6e runner selects FP8/512 while
   leaving the stored checkpoint in MXFP4.

## 20B-first validation gates

The trainer portion was live-validated on 2026-09-02 with
`openai/gpt-oss-20b`, v6e-16, TP=8/FSDP=2, rank 32, two 256-token rows, one
extra consecutive update, and one checkpoint replay. The gate passed with
nonzero sparse expert gradients, exact restore/replay deltas (`0.0`), gradient
norm `19.378279` reproduced after restore, and flat post-pass HBM near 3.2 GB
per device (13.7 GB observed peak). This validates MaxText sparse training,
Qwix attention/router factors, optimizer state, and trainer checkpoint resume.
Together with the vLLM result above, training and inference are independently
accepted; a live colocated weight-sync loop and 120B capacity are still
unproven.

The durable acceptance record is
`gs://sk7524-tinker-tpu-us-east5/v6e-smoke-results/gptoss20b-sparse-lora-tp8-fsdp2-r32-s256-0e5e43a3-d388c5478.json`.
The corresponding immutable SkyRL source bundle is
`gs://sk7524-tinker-tpu-us-east5/code-bundles/gptoss20b-sparse-lora-v6e-live-0e5e43a3-maxtext-d388c5478.tar.gz`
with SHA-256
`824c695c7b101b899f7b68ce0cfa50ee2341b62eb6fb157e45294808249b7777`.

1. CPU unit tests cover multi-slot factor layout, transactional slot updates,
   immutable base tensors, and mixed-slot pre-activation/pre-reduction
   numerical ordering.
2. The v6e-8 gate compiles `openai/gpt-oss-20b`, installs fixed rank-1 test
   factors into rank-32 banks, and proves zero-adapter/base parity.
3. Deterministic nonzero expert and router factors both change live outputs.
4. Repeated concurrent base/zero/A/B batches preserve adapter-family
   separation; forced LRU reuse clears stale expert state, and final clear
   restores the initial base fingerprints with no drift.
5. Benchmark prefill/decode and HBM. The first implementation adds two shared
   dense shrink matmuls, three skinny grouped matmuls, and one shared dense
   expand matmul per layer. A later Pallas kernel can fuse each low-rank A-to-B
   chain if profiling says launch cost matters.
6. Run a short SkyRL/Tunix rollout-training sync loop, then a checkpoint resume.

Only after those gates pass should the same path be enabled for 120B. The 120B
factor footprint grows with 128 experts and 36 layers (about 2.4 GiB in BF16 at
rank 32 before padding), so EP sharding and upload time must be measured
explicitly.

## Code map

- `tpu_inference/layers/common/moe_lora.py`: factor pytree contract.
- `tpu_inference/layers/common/fused_moe_gmm.py`: correct LoRA math and TP/EP
  shard-map specs.
- `tpu_inference/layers/vllm/quantization/mxfp4.py`: fixed-shape factor buffer
  allocation and MXFP4 wiring.
- `tpu_inference/runner/tpu_runner.py`: validated replacement/clear operation.
- `tpu_inference/worker/tpu_worker.py`: `set_moe_lora_factors` RPC.
- SkyRL MaxText fork, `src/maxtext/layers/moe.py`: sparse training math inside
  routed-token GMM execution.
- SkyRL MaxText fork, `src/maxtext/utils/lora_utils.py`: scanned factor creation
  and logical sharding.
- `skyrl/backends/tunix_backend.py`: MaxText/Qwix export, with router in PEFT.
- `tpu/vllm_tpu_server.py`: upload orchestration.
