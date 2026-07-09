# SkyRLTpu Takeover Notes

Last checked: 2026-07-09 17:00 ET.

This worktree is `/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-colocated-vllm` on branch
`agent/colocated-vllm-one-host`. It is a SkyRLTpu fork checkout with:

- `origin`: `https://github.com/SachinKonan/SkyRLTpu.git`
- `upstream`: `https://github.com/NovaSky-AI/SkyRL.git`

## Current TPU State

The active colocated server is:

```text
project: vision-mix
zone: us-east5-a
tpu name: sk7524-coloc-vllm-qwen35-4b-v5p32-r5-east5a_spot
shape: v5p-32
state: READY
health: HEALTHY
```

Important naming caveat: the TPU VM name contains `qwen35-4b`, but the running
server is serving `Qwen/Qwen3.5-9B`.

Worker layout:

```text
worker 0: Tinker API / SkyRL JAX coordinator, internal 10.202.0.69, external 8.234.36.173
worker 1: SkyRL JAX worker,                internal 10.202.0.70, external 34.162.175.61
worker 2: vLLM TPU server,                 internal 10.202.0.57, external 8.234.63.8
worker 3: vLLM TPU server,                 internal 10.202.0.84, external 8.234.43.25
```

The current deployment uses two independent vLLM servers, one on worker 2 and
one on worker 3. SkyRL round-robins requests across:

```text
http://10.202.0.57:8001,http://10.202.0.84:8001
```

This is not yet vLLM-native data parallel routing.

Active remote tmux sessions:

```text
worker 0: skyrl-tinker
worker 1: skyrl-tinker-worker-1
worker 2: vllm-tpu
worker 3: vllm-tpu
```

Local tunnel tmux session:

```text
skyrl-math-tinker-tunnel-v5p32-r5-vllm9b-replicas
```

Local Tinker API URL:

```text
http://127.0.0.1:18025
```

Health check:

```bash
curl -fsS http://127.0.0.1:18025/api/v1/get_server_capabilities
```

Expected response:

```json
{"supported_models":[{"model_name":"Qwen/Qwen3.5-9B"}]}
```

## Login And Logs

SSH to worker 0:

```bash
gcloud alpha compute tpus tpu-vm ssh sk7524_princeton_edu@sk7524-coloc-vllm-qwen35-4b-v5p32-r5-east5a_spot \
  --project=vision-mix \
  --zone=us-east5-a \
  --worker=0 \
  --ssh-key-file="$HOME/.ssh/jobman_tpu_ed25519" \
  --quiet
```

Tail the Tinker API log:

```bash
gcloud alpha compute tpus tpu-vm ssh sk7524_princeton_edu@sk7524-coloc-vllm-qwen35-4b-v5p32-r5-east5a_spot \
  --project=vision-mix \
  --zone=us-east5-a \
  --worker=0 \
  --ssh-key-file="$HOME/.ssh/jobman_tpu_ed25519" \
  --quiet \
  --command='tail -f ~/skyrl-logs/tinker-api.log'
```

Tail the train worker log:

```bash
gcloud alpha compute tpus tpu-vm ssh sk7524_princeton_edu@sk7524-coloc-vllm-qwen35-4b-v5p32-r5-east5a_spot \
  --project=vision-mix \
  --zone=us-east5-a \
  --worker=1 \
  --ssh-key-file="$HOME/.ssh/jobman_tpu_ed25519" \
  --quiet \
  --command='tail -f ~/skyrl-logs/tinker-worker-1.log'
```

Tail vLLM logs:

```bash
for w in 2 3; do
  gcloud alpha compute tpus tpu-vm ssh sk7524_princeton_edu@sk7524-coloc-vllm-qwen35-4b-v5p32-r5-east5a_spot \
    --project=vision-mix \
    --zone=us-east5-a \
    --worker="$w" \
    --ssh-key-file="$HOME/.ssh/jobman_tpu_ed25519" \
    --quiet \
    --command='tail -f ~/skyrl-logs/vllm-tpu.log'
done
```

Recreate the local tunnel manually if the local tmux session dies:

```bash
ssh -i "$HOME/.ssh/jobman_tpu_ed25519" \
  -L 127.0.0.1:18025:127.0.0.1:8000 \
  sk7524_princeton_edu@8.234.36.173
```

## Jobman Naming

Current reliable source of truth is live GCP state plus the tunnel/API health
checks above. `jobman list` most recently displayed the colocated v5p-32 as job
ID `000012` with name:

```text
sk7524-coloc-vllm-qwen35-4b-v5p32-r5-east5a_spot
```

The `r5` suffix is the current active v5p-32 colocated attempt. There are other
east5-a jobs/resources from separate experiments, including:

```text
sk7524-math-qwen35-9b-v5p16-autoresume-east5a_1
sk7524-math-qwen35-9b-v5p16-r11-east5a_spot
```

Do not clean these up based only on local `jobs/sk7524/<id>` files; some of the
repo-local Jobman state is stale. For example, the local
`jobs/sk7524/000012/config.yaml` still described an older v5p-64 attempt even
though `jobman list` showed the live colocated v5p-32. Before deleting anything,
run:

```bash
uv run jobman list
gcloud alpha compute tpus tpu-vm list --project=vision-mix --zone=us-east5-a
```

## Launch Shape

The active server was launched with the colocated launcher:

```bash
tpu/start_colocated_vllm_tinker.sh
```

Key settings for the active deployment:

```text
MODEL_NAME=Qwen/Qwen3.5-9B
TRAIN_WORKERS=0,1
VLLM_WORKERS=2,3
TP_SIZE=4
FSDP_SIZE=2
TRAIN_MICRO_BATCH_SIZE=8
SAMPLE_MAX_NUM_SEQUENCES=256
VLLM_MAX_CONCURRENT_REQUESTS=256
VLLM_MAX_NUM_SEQS=256
VLLM_TP_SIZE=4 per vLLM server
VLLM_MAX_MODEL_LEN=2048
VLLM_RAY_EXECUTOR=0
VLLM_TPU_BACKEND_TYPE=torchax
```

The remote SkyRLTpu checkout is synchronized from this local worktree into
`/home/sk7524_princeton_edu/SkyRLTpu` on workers 0-3. Runtime checkpoint and LoRA
adapter paths are under the GCS-mounted directory:

```text
/home/sk7524_princeton_edu/gcs/skyrl-checkpoints
/home/sk7524_princeton_edu/gcs/skyrl-lora-models
```

## Code Changes In This Worktree

Backend / sampler:

- `skyrl/backends/jax.py` adds vLLM LoRA unload and retry config fields.
- `skyrl/backends/vllm_sampling.py` supports comma-separated vLLM URLs, bounded
  LoRA load retries, unloading the previous adapter before loading a new one,
  and clearer vLLM request errors.
- `tests/backends/test_vllm_sampling_client.py` covers LoRA unload and retry
  behavior.

TPU launchers:

- `tpu/start_vllm_tpu.sh` now supports multiple vLLM workers, TorchAX by default,
  optional Ray executor setup, TPU process address plumbing, `SKIP_JAX_PRECOMPILE`,
  and duplicate per-worker vLLM engines when `VLLM_RAY_EXECUTOR=0`.
- `tpu/apply_vllm_tpu_lora_patch.sh` patches the installed TPU vLLM package for
  runtime LoRA support and ensures `SKIP_JAX_PRECOMPILE` is included in the TPU
  platform env allowlist.
- `tpu/start_colocated_vllm_tinker.sh` is the new one-command launcher for a
  disjoint train/vLLM worker split on one TPU VM.
- `tpu/README.md` documents the vLLM TPU LoRA/TorchAX setup.

Current limitation:

- The active deployment uses SkyRL-side round-robin across two independent vLLM
  servers. It does not yet use vLLM-native data parallel routing.
- The client still sends one completion request per environment sample. The next
  optimization target is to use grouped completions for `MAX_TURNS=1`, so one
  prompt can request `group_size` completions through vLLM's `n` parameter.

## Current Results

Completed no-eval smoke benchmark:

```text
client session: skyrl-math-rl-qwen35-9b-v5p32-r5-vllm-replicas-noeval
client log: /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-colocated-vllm/runs/math_rl/math-rl-qwen35-9b-v5p32-r5-vllm-replicas-noeval-2026-07-09-16-06.log
metrics: /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-colocated-vllm/runs/math_rl/math-Qwen-Qwen3.5-9B-32rank-2e-05lr-16group-64batch-importance_sampling-seed0-1024tok-v5p32-r5-vllm-replicas-noeval-2026-07-09-16-06/metrics.jsonl
```

Client command summary:

```text
model_name=Qwen/Qwen3.5-9B
groups_per_batch=64
group_size=16
max_tokens=1024
lora_rank=32
learning_rate=2e-5
max_steps=2
eval_every=0
```

Step metrics:

| step | sampling sec | generated tokens | sampling tok/s | train step sec | total sec | correct |
| ---: | -----------: | ---------------: | -------------: | -------------: | --------: | ------: |
| 0 | 262.990 | 1,034,313 | 3,932.9 | 711.195 | 1,019.648 | 0.2461 |
| 1 | 155.894 | 1,010,640 | 6,482.9 | 787.153 | 990.528 | 0.2852 |

The run completed successfully and saved final weights:

```text
tinker://model_c3ebfef2/weights/final
tinker://model_c3ebfef2/final
```

The train step is slower than the older v5p-16 all-Tinker baseline because this
experiment only gives training workers 0-1 and reserves workers 2-3 for vLLM.
The sampling path is functional and LoRA updates were observed on both vLLM
servers via `/v1/unload_lora_adapter` followed by `/v1/load_lora_adapter`.

## Validation

Local validation already run:

```bash
bash -n tpu/start_colocated_vllm_tinker.sh
bash -n tpu/start_vllm_tpu.sh
uv run --extra dev python -m pytest tests/backends/test_vllm_sampling_client.py
```

The pytest target passed with `5 passed`.
