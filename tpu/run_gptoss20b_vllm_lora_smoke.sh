#!/usr/bin/env bash
# Run the live GPT-OSS 20B MXFP4 + split-LoRA acceptance gate on one v6e-8.
set -euo pipefail

: "${SMOKE_RESULT_GCS:?canonical result URI is required}"
: "${SMOKE_LOG_GCS:?regional server-log URI is required}"
: "${SMOKE_FAILURE_GCS_PREFIX:?regional failure prefix is required}"

REPO="${SKYRL_REPO_DIR:-$HOME/SkyRLTpu-gptoss20b-vllm}"
VENV="${VLLM_VENV:-$HOME/.venvs/vllm-tpu-gptoss20b-lora}"
MODEL="${MODEL_NAME:-openai/gpt-oss-20b}"
PORT="${VLLM_PORT:-8001}"
ZONE="${ZONE:-unknown}"
TPUINF_COMMIT="${TPU_INFERENCE_COMMIT:-b9e4024b5624fe74d7486c1b8dc34b1ce45c8aaa}"
RESULT="${SMOKE_RESULT_LOCAL:-/tmp/gptoss20b-vllm-lora-result.json}"
LOG="${SMOKE_LOG_LOCAL:-$HOME/skyrl-logs/gptoss20b-vllm-lora.log}"
HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"

export PATH="$HOME/.local/bin:$PATH"
mkdir -p "$(dirname "$LOG")" "$HF_HOME/hub" "$(dirname "$VENV")"

if gcloud storage objects describe "$SMOKE_RESULT_GCS" >/dev/null 2>&1; then
  echo "canonical GPT-OSS 20B vLLM acceptance result already exists"
  exit 0
fi

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  if [ -f "$LOG" ]; then
    gcloud storage cp "$LOG" "$SMOKE_LOG_GCS" >/dev/null 2>&1 || true
  fi
  if [ "$status" -ne 0 ] && [ -f "$RESULT" ]; then
    local stamp
    stamp=$(date -u +%Y%m%dT%H%M%SZ)
    gcloud storage cp "$RESULT" "${SMOKE_FAILURE_GCS_PREFIX%/}/${stamp}.json" >/dev/null 2>&1 || true
  fi
  pkill -TERM -u "$USER" -f '[v]llm_tpu_server.py.*gpt-oss-20b' 2>/dev/null || true
  pkill -TERM -u "$USER" -f '[V]LLM::EngineCore' 2>/dev/null || true
  exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -x "$VENV/bin/python" ] || uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" "vllm-tpu==0.23.0" "transformers==5.8.0"
uv pip install --python "$VENV/bin/python" --no-deps --force-reinstall \
  "tpu-inference @ git+https://github.com/SachinKonan/tpu-inference.git@${TPUINF_COMMIT}"

"$VENV/bin/python" - <<'PY'
import inspect
from tpu_inference.layers.vllm.quantization.mxfp4 import VllmMxfp4MoEMethod
from tpu_inference.worker.tpu_worker import TPUWorker

assert hasattr(TPUWorker, "set_moe_lora_factors")
assert "_create_lora_buffers" in inspect.getsource(VllmMxfp4MoEMethod)
assert "max_loras" in inspect.getsource(VllmMxfp4MoEMethod._create_lora_buffers)
print("pinned GPT-OSS MXFP4 multi-LoRA runtime verified")
PY

export HF_HOME
export TRANSFORMERS_CACHE="$HF_HOME/hub"
export MODEL_IMPL_TYPE=vllm
export TPU_BACKEND_TYPE=torchax
export SKIP_JAX_PRECOMPILE=1
export USE_MOE_EP_KERNEL=0
# v6e has FP8 GMM support but not native FP4 GMM support.  The checkpoint
# remains MXFP4 on disk; its experts are dequantized once at load time and
# requantized to FP8 for execution on v6e.
export MOE_REQUANTIZE_WEIGHT_DTYPE="${MOE_REQUANTIZE_WEIGHT_DTYPE:-fp8}"
export MOE_REQUANTIZE_BLOCK_SIZE="${MOE_REQUANTIZE_BLOCK_SIZE:-512}"
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
export VLLM_LORA_RESOLVER_CACHE_DIR="$HOME/skyrl-local-loras"
mkdir -p "$VLLM_LORA_RESOLVER_CACHE_DIR"

pkill -TERM -u "$USER" -f '[v]llm_tpu_server.py.*gpt-oss-20b' 2>/dev/null || true
pkill -TERM -u "$USER" -f '[V]LLM::EngineCore' 2>/dev/null || true

"$VENV/bin/python" "$REPO/tpu/vllm_tpu_server.py" "$MODEL" \
  --skyrl-lora-dir "$VLLM_LORA_RESOLVER_CACHE_DIR" \
  --served-model-name "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --tensor-parallel-size 8 \
  --max-model-len 256 \
  --max-num-batched-tokens 256 \
  --max-num-seqs 8 \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 32 \
  --gpu-memory-utilization 0.80 \
  --download-dir "$HF_HOME/hub" \
  >"$LOG" 2>&1 &
server_pid=$!

ready=0
for _ in $(seq 1 720); do
  if ! kill -0 "$server_pid" 2>/dev/null; then
    echo "vLLM server exited before readiness" >&2
    tail -200 "$LOG" >&2 || true
    exit 1
  fi
  if curl -fsS -m 5 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
    ready=1
    break
  fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "vLLM readiness timed out" >&2; tail -200 "$LOG" >&2; exit 1; }

repo_commit=$(cat "$REPO/.tpuswarm-bundle-manifest" 2>/dev/null | sed -n 's/^parent_commit=//p' | head -1)
repo_commit="${repo_commit:-unknown}"
"$VENV/bin/python" "$REPO/tpu/gptoss20b_vllm_lora_smoke.py" \
  --base-url "http://127.0.0.1:${PORT}" \
  --result "$RESULT" \
  --zone "$ZONE" \
  --repo-commit "$repo_commit" \
  --tpu-inference-commit "$TPUINF_COMMIT"

# The three regional jobs share one canonical success object. Conditional
# creation prevents a slower second winner from replacing the first proof.
gcloud storage cp --if-generation-match=0 "$RESULT" "$SMOKE_RESULT_GCS" \
  || gcloud storage objects describe "$SMOKE_RESULT_GCS" >/dev/null

echo "GPT-OSS 20B vLLM MXFP4 + LoRA acceptance passed: $SMOKE_RESULT_GCS"
