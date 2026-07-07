# SkyRLTpu Runtime Notes

This directory contains the TPU launch wrappers used for the SkyRLTpu checkout.

## vLLM TPU LoRA

`start_vllm_tpu.sh` starts an external vLLM TPU server for sampling. It pins
`vllm-tpu==0.23.0`, forces `MODEL_IMPL_TYPE=vllm`, and applies the local
`TPUWorker` LoRA patch before launching the server. The forced model
implementation is required for Qwen LoRA serving; the default TPU model
implementation takes the JAX path, where runtime LoRA loading is not wired up
for this workflow.

The source baseline is tracked as the `third_party/tpu-inference` submodule at
the upstream `releases/v0.23.0` commit. Our local patch is stored in
`third_party/patches/tpu-inference-tpu-worker-lora-forwarders.patch` and can be
applied to an installed environment with:

```bash
PYTHON=/path/to/venv/bin/python tpu/apply_vllm_tpu_lora_patch.sh
```

The patch only adds four `TPUWorker` forwarders:

- `add_lora`
- `remove_lora`
- `list_loras`
- `pin_lora`

Those methods delegate to `TPUModelRunner`, which already mixes in vLLM's LoRA
runner support.

## Adapter Format

SkyRL's JAX backend saves sampler checkpoints through `save_lora_checkpoint`.
Those archives already contain vLLM-compatible keys like
`base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight`, without the
PEFT `.default` suffix that vLLM TPU rejects.

Use versioned LoRA names when reloading updated adapters into a long-running
vLLM server. Reusing the same LoRA name after changing the files can leave
prefix-cache entries associated with the old adapter state.

## Typical Launch

```bash
tpu/start_vllm_tpu.sh
```

For the current east-zone two-slice setup, point SkyRL training at the vLLM
server with:

```bash
INFERENCE_BACKEND=vllm \
VLLM_BASE_URL=http://<vllm-internal-ip>:8001 \
tpu/start_skyrl_tinker.sh
```
