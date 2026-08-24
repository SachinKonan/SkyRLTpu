#!/usr/bin/env bash
# v6e-8 variant of tpu/pallas_arena/probe/serve_vllm.sh: bring up ONE sampling-only
# vLLM server on a 4-CHIP SUBSET of the single v6e-8 host (TP=4), so two instances
# (qwen on chips 0-3, gemma on chips 4-7) share the host without contention.
#
# Differences from the v5p probe script (deliberate, minimal):
#   * TPU_VISIBLE_CHIPS is REQUIRED (the v5p script unsets it to take all 4 host
#     chips; here each instance must see only its half of the 8 chips).
#   * TPU_PROCESS_PORT is per-instance so the two standalone libtpu runtimes
#     cannot collide on the default coordinator port.
#   * pkill/tmux are scoped to THIS instance (INSTANCE name) -- the v5p script
#     pkills every "vllm serve" on the host, which would murder the other half.
set -euo pipefail

MODEL="${MODEL:?MODEL required}"
PORT="${PORT:-8001}"
INSTANCE="${INSTANCE:?INSTANCE required (e.g. qwen35 / gemma4)}"
CHIPS="${CHIPS:?CHIPS required (e.g. 0,1,2,3)}"
TPU_PORT="${TPU_PORT:-8476}"
BIND_HOST="${BIND_HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
HF_CACHE_GCS="${HF_CACHE_GCS:-}"
XLA_CACHE_GCS="${XLA_CACHE_GCS:-}"
LOG="${LOG:-$HOME/vllm-${INSTANCE}.log}"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
command -v tmux >/dev/null 2>&1 || sudo apt-get install -y -qq tmux

# Per-instance venv: the gemma transformers pin (5.14.0) must not downgrade the
# qwen instance's env mid-flight on a shared host.
VENV="$HOME/.venvs/vllm-tpu-${INSTANCE}"
if ! "$VENV/bin/python" -c "import vllm" >/dev/null 2>&1; then
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" "vllm-tpu==0.23.0"
  uv pip install --python "$VENV/bin/python" hf_transfer || true
fi
if [ -n "${EXTRA_PIP:-}" ]; then
  echo "applying extra pins: $EXTRA_PIP"
  uv pip install --python "$VENV/bin/python" $EXTRA_PIP || echo "extra pin failed (continuing)"
fi

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME/hub"
if [ -n "$HF_CACHE_GCS" ]; then
  echo "restoring HF cache from ${HF_CACHE_GCS}"
  gsutil -m -q rsync -r "$HF_CACHE_GCS" "$HF_HOME/hub" 2>/dev/null \
    && echo "HF cache restored" || echo "HF cache restore failed (will download from the hub)"
fi

# Shared local XLA cache (content-addressed, additive; both instances write here).
export VLLM_XLA_CACHE_PATH="$HOME/vllm-xla-cache-local"
mkdir -p "$VLLM_XLA_CACHE_PATH"
if [ -n "$XLA_CACHE_GCS" ]; then
  gsutil -m -q rsync -r "$XLA_CACHE_GCS" "$VLLM_XLA_CACHE_PATH" 2>/dev/null \
    && echo "XLA cache restored from ${XLA_CACHE_GCS}" || echo "XLA cache restore failed (will compile)"
fi

# scoped kill: this instance only
tmux kill-session -t "vllm-${INSTANCE}" >/dev/null 2>&1 || true
pkill -f "vllm-tpu-${INSTANCE}.*vllm serve" >/dev/null 2>&1 || true
sleep 3
rm -f "$LOG"

tmux new-session -d -s "vllm-${INSTANCE}" \
  "MODEL_IMPL_TYPE=vllm TPU_BACKEND_TYPE=torchax SKIP_JAX_PRECOMPILE=1 CLOUD_TPU_TASK_ID=0 \
   TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
   TPU_VISIBLE_CHIPS='$CHIPS' TPU_PROCESS_PORT=$TPU_PORT \
   HF_HOME='$HF_HOME' HF_HUB_ENABLE_HF_TRANSFER=1 VLLM_XLA_CACHE_PATH='$VLLM_XLA_CACHE_PATH' \
   '$VENV/bin/vllm' serve '$MODEL' \
     --served-model-name '$MODEL' \
     --host ${BIND_HOST:-127.0.0.1} --port $PORT \
     --tensor-parallel-size 4 \
     --max-model-len $MAX_MODEL_LEN \
     --max-num-seqs $MAX_NUM_SEQS \
     --max-num-batched-tokens 4096 \
     --download-dir '$HF_HOME/hub' \
     2>&1 | tee -a '$LOG'"

echo "serve_v6e: launched ${MODEL} (instance ${INSTANCE}, chips ${CHIPS}) on port ${PORT} (log ${LOG})"
