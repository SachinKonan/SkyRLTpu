# EasyDeL Tinker Backend

SkyRL can use EasyDeL as an alternative TPU learner and eSurge as its
co-located sampling frontend. This path is selected with `--backend easydel`.
It does not replace the existing JAX learner or external vLLM frontend.

## Install

Install the Tinker API and the pinned EasyDeL backend together:

```bash
uv sync --extra tpu --extra tinker --extra easydel
```

The EasyDeL revision is pinned in `pyproject.toml` so model conversion,
sharding, eLoRA, and eSurge use the versions covered by the backend tests.

## Single Host

The backend accepts the normal SkyRL Tinker requests. Sampling is handled by
eSurge inside the backend, so do not set `--external-inference-url`.

```bash
uv run --extra tpu --extra tinker --extra easydel \
  python -m skyrl.tinker.api \
  --base-model Qwen/Qwen3.5-9B \
  --backend easydel \
  --backend-config '{
    "model_name_or_path": "EasyDeL/Qwen3.5-9B",
    "from_torch": false,
    "tensor_parallel_size": -1,
    "sequence_parallel_size": 1,
    "train_micro_batch_size": 1,
    "sample_max_model_len": 131072,
    "sample_max_num_sequences": 32,
    "sample_hbm_utilization": 0.8
  }'
```

`base_model` remains the Tinker API identity. `model_name_or_path` selects the
checkpoint EasyDeL actually loads. Qwen 3.5 defaults to the native converted
EasyDeL checkpoint when the override is omitted.

## Multi-Host v5p-16

A v5p-16 queued resource has two TPU VM workers. Start the non-coordinator
worker first, then start the Tinker API on worker 0. Both commands must use the
same coordinator address and process count. Leave `JAX_PLATFORMS` unset on
both workers so PJRT is initialized only after the hosts join JAX distributed.

eSurge's multi-host mode also needs one service name that resolves both TPU VM
workers. Kubernetes provides this through a headless service. With Jobman TPU
VMs, add the two per-worker addresses under a shared name in `/etc/hosts` on
both workers (the smoke template automates this):

```bash
ESURGE_SERVICE_NAME=skyrl-esurge-workers
for worker in 0 1; do
  host="${TPU_VM_NAME}-w-${worker}"
  ip="$(getent ahostsv4 "$host" | awk 'NR == 1 { print $1 }')"
  printf '%s %s\n' "$ip" "$ESURGE_SERVICE_NAME"
done | sudo tee -a /etc/hosts
```

Worker 1:

```bash
env -u JAX_PLATFORMS ENABLE_DISTRIBUTED_INIT=0 \
  uv run --extra tpu --extra tinker --extra easydel \
  python -m skyrl.backends.easydel \
    --coordinator-address "${WORKER0_INTERNAL_IP}:12355" \
    --num-processes 2 \
    --process-id 1
```

Worker 0:

```bash
env -u JAX_PLATFORMS ENABLE_DISTRIBUTED_INIT=0 \
  uv run --extra tpu --extra tinker --extra easydel \
  python -m skyrl.tinker.api \
  --base-model Qwen/Qwen3.5-9B \
  --backend easydel \
  --backend-config "$(python - <<PY
import json

print(json.dumps({
    'model_name_or_path': 'EasyDeL/Qwen3.5-9B',
    'from_torch': False,
    'coordinator_address': '${WORKER0_INTERNAL_IP}:12355',
    'num_processes': 2,
    'sample_distributed_service_name': '${ESURGE_SERVICE_NAME}',
    'sample_distributed_hosts': [
        '${TPU_VM_NAME}-w-0',
        '${TPU_VM_NAME}-w-1',
    ],
    'tensor_parallel_size': 4,
    'sequence_parallel_size': 2,
    'train_micro_batch_size': 1,
    'sample_max_model_len': 131072,
    'sample_max_num_sequences': 32,
    'sample_hbm_utilization': 0.8,
}))
PY
)"
```

The `(tp=4, sp=2)` mesh consumes all eight v5p-16 devices. EasyDeL owns JAX
distributed initialization. SkyRL commands use a separate CPU TCP channel,
while eSurge uses its own lockstep leader/worker control plane for TPU launches;
do not set `use_ray` for this backend.

The ordered host list is required for Jobman. Upstream eSurge assigns ranks by
sorting service IPs, but Cloud TPU worker IP order is not guaranteed to match
the `w-0`, `w-1` process order. The backend supplies the explicit order to
eSurge discovery while retaining the normal DNS service path for Kubernetes.

## Behavior

- Training uses EasyDeL model conversion, eLoRA, rematerialization, sequence
  sharding, and chunked LM-head log-probability computation.
- Sampling uses a merged policy view in eSurge and enables prefix caching.
- If eSurge does not return a token log-probability, SkyRL teacher-forces the
  generated token through the same policy so PPO receives exact behavior
  log-probabilities. This preserves correctness at the cost of an extra scoring
  pass for affected generations.
- Tinker checkpoints include LoRA parameters, optimizer state, and step. The
  existing autoresume controller can continue to persist those checkpoint
  archives to its shared GCS path.
- Multi-host restore is lockstep. Stage the learner and sampler archives at
  the same local path on every TPU VM before issuing `load_state` or loading a
  sampler archive; the Jobman smoke template does this before worker startup.
- EasyDeL currently supports policy models only. Critic models and
  `ppo_critic` remain on the SkyRL-Train backend.

## Tests

Fast compatibility tests:

```bash
uv run --extra easydel --extra tinker --extra dev \
  python -m pytest tests/backends/test_easydel_backend.py -q
```

Model-backed learner, checkpoint, eSurge, and RL lifecycle tests:

```bash
SKYRL_RUN_EASYDEL_INTEGRATION=1 \
  uv run --extra easydel --extra tinker --extra dev \
  python -m pytest tests/backends/test_easydel_backend.py -q
```

The numerical parity test maps the existing JAX backend's fused,
group-interleaved, max-rank-padded QKV and gate/up adapters to EasyDeL's
separate requested-rank tensors. It then compares token log-probabilities,
losses, mapped LoRA gradients and gradient norms, and mapped parameters after
one identical AdamW step.

Run the public Tinker client against real API and EasyDeL engine processes:

```bash
SKYRL_RUN_EASYDEL_API_INTEGRATION=1 \
  uv run --extra easydel --extra tinker --extra dev \
  python -m pytest tests/tinker/test_api.py \
    -k easydel_real_process_training_and_sampling_workflow -q
```

Run the v5p-16 lifecycle smoke, or resume its persisted sampler/optimizer
state and execute the progressive long-context RL ladder:

```bash
RUN_ID=easydel-qwen35-9b-lifecycle \
  ./tpu/submit_easydel_tpu_smoke.sh

RUN_ID=easydel-qwen35-9b-resume-longctx \
LONG_CONTEXT_LENGTHS=2048,8192,32768,45000 \
RESUME_RESULT_PREFIX=gs://BUCKET/path/to/passing-lifecycle/results \
SMOKE_POLL_ATTEMPTS=1800 \
  ./tpu/submit_easydel_tpu_smoke.sh

# Skip eSurge lifecycle compilation when measuring only the learner. This
# restores learner/optimizer state directly and is suitable for preemptible
# progressive context tests.
RUN_ID=easydel-qwen35-9b-longctx-only \
LONG_CONTEXT_ONLY=1 \
LONG_CONTEXT_LENGTHS=2048,8192,32768,45000 \
RESUME_RESULT_PREFIX=gs://BUCKET/path/to/passing-lifecycle/results \
  ./tpu/submit_easydel_tpu_smoke.sh
```

The long-context harness runs a behavior-log-probability pass, PPO
forward/backward, and AdamW update twice at every length. It records first
compile/step time, warmed step time, gradient norm, and per-device HBM stats in
`long-context-progress.json` after every successful length, so a later failure
does not discard earlier measurements.

## Validated v5p-16 Results

The Qwen3.5-9B TP4/SP2 release gates passed on July 10, 2026. The lifecycle
smoke generated two sequences, performed a PPO update with a nonzero gradient,
persisted learner and sampler state, refreshed the retained eSurge executable,
and generated again. A newly provisioned two-host slice then restored optimizer
step 1, sampled successfully, advanced to step 2 with gradient norm `0.90234`,
saved both checkpoint forms, sampled after the update, shut down both JAX hosts,
and was deleted cleanly.

The progressive learner test used exact behavior-log-probability scoring and
two PPO/AdamW updates at each context length. Timings are wall-clock seconds;
HBM is the maximum `peak_bytes_in_use` reported by a local TPU device.

| Tokens | First update | Warmed update | Gradient norms | Peak HBM |
|---:|---:|---:|---:|---:|
| 2,048 | 275.80 | 266.24 | 1.39844 / 0.76562 | 8.67 GB |
| 8,192 | 240.41 | 3.23 | 0.04858 / 0.02966 | 9.51 GB |
| 32,768 | 307.93 | 13.30 | 0.00537 / 0.00439 | 10.36 GB |
| 45,000 | 399.75 | 22.72 | 0.00313 / 0.00308 | 11.22 GB |

The run reported `mesh={tp: 4, sp: 2}`,
`lmhead_token_chunk_size=128`, and `lmhead_vocab_chunk_size=32768`. The 2K
measurement includes one-time restored-state/device initialization in both
passes; use the 8K-and-above warmed measurements for steady-state scaling.
