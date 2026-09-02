# GPT-OSS LoRA on TPU: immutable MXFP4 experts

Status: training and inference implementations are complete for the
single-active-adapter GMM path; CPU numeric/integration tests are included. The
20B trainer acceptance gate passes on a four-host v6e-16 with TP=8/FSDP=2.
Colocated vLLM weight-sync validation and the 120B v6e-32 gate remain before a
long RL run.

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
| gate (`wi_0`) | `(H, R)` | `(R, E, I)` | A `(Hpad,Rmax)`, B `(E,Rmax,Ipad)` |
| up (`wi_1`) | `(H, R)` | `(R, E, I)` | A `(Hpad,Rmax)`, B `(E,Rmax,Ipad)` |
| down (`wo`) | `(E, I, R)` | `(R, H)` | A `(E,Ipad,Rmax)`, B `(Rmax,Hpad)` |
| router | `(H, R)` | `(R, E)` | ordinary PEFT `mlp.router` tensors |

Rank is padded to vLLM's `max_lora_rank`, so uploads replace values without
changing the JAX pytree or causing a rank-specific compilation.

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
   shardings. MXFP4 blocks/scales are loaded and requantized exactly as before.
2. The trainer exports attention and router weights in
   `adapter_model.safetensors`. Expert factors remain in
   `moe_lora.safetensors` with `gptoss-moe-lora/v1` metadata.
3. The upload endpoint calls `set_moe_lora_factors` on every worker. The RPC
   reads the worker-local safetensors file, validates shapes/ranks, transposes
   the serialized `(R,E,I)` B tensors to the GMM's `(E,R,I)` layout, pads them,
   and replaces the factor buffers.
4. vLLM loads the PEFT adapter normally for attention and the BF16 router.
5. Moving to an adapter without expert factors explicitly zeroes all expert
   buffers.

V1 intentionally supports one globally active expert adapter. vLLM can still
cache multiple ordinary PEFT adapters, but per-request multiplexing of expert
factors is out of scope. RL weight sync uses one current policy, so this removes
adapter/expert double grouping from the first implementation.

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

The first acceptance run should use four equal 1,024-token rows, which matches
FSDP=4 and avoids conflating sparse-LoRA correctness with long-context memory.
After zero-adapter parity and a nonzero-gradient update pass, increase sequence
length and token budget independently. Do not start at the production context
length: compile time and activation HBM, rather than model weights, become the
dominant uncertainty.

## 20B-first validation gates

The trainer portion was live-validated on 2026-09-02 with
`openai/gpt-oss-20b`, v6e-16, TP=8/FSDP=2, rank 32, two 256-token rows, one
extra consecutive update, and one checkpoint replay. The gate passed with
nonzero sparse expert gradients, exact restore/replay deltas (`0.0`), gradient
norm `19.378279` reproduced after restore, and flat post-pass HBM near 3.2 GB
per device (13.7 GB observed peak). This validates MaxText sparse training,
Qwix attention/router factors, optimizer state, and trainer checkpoint resume;
it does not yet validate vLLM adapter upload or 120B capacity.

1. Run CPU unit tests for factor layout, replacement/clear semantics, immutable
   base tensors, and pre-activation/pre-reduction numerical ordering.
2. On one v6e-8, compile native `openai/gpt-oss-20b` MXFP4 with rank-32 zero
   buffers and prove zero-adapter logits match the base path.
3. Install deterministic random factors and compare layer outputs/logits with
   a dense BF16 reference implementation on fixed prompts.
4. Replace adapter A with B and then clear it; verify exact B-only/base outputs
   with no drift across repeated swaps.
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
