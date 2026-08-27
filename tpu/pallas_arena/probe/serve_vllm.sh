#!/usr/bin/env bash
# Bring up ONE sampling-only vLLM server on ONE v5p host (4 chips, TP=4).
# Runs ON the TPU VM. Idempotent.
#
# Deliberately NOT tpu/start_vllm_tpu.sh: that script exists to serve a
# TRAINING run -- it installs a tpu-inference fork purely for LoRA
# forwarders, asserts those forwarders exist, and passes --enable-lora.
# The probe pushes no adapters and trains nothing, so stock vllm-tpu is both
# sufficient and one git-clone-and-build faster. The env vars below are the
# ones that script sets and that actually matter off the LoRA path.
set -euo pipefail

MODEL="${MODEL:?MODEL required}"
PORT="${PORT:-8001}"
# Default stays loopback (the pallas probe reaches it locally). The llama-farm sets
# BIND_HOST=0.0.0.0 so neuronic can reach it directly over the existing
# strategist-vllm-ingress firewall rule (128.112.0.0/16 -> tcp:8001-8012).
BIND_HOST="${BIND_HOST:-127.0.0.1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
HF_CACHE_GCS="${HF_CACHE_GCS:-}"
XLA_CACHE_GCS="${XLA_CACHE_GCS:-}"
LOG="${LOG:-$HOME/vllm-probe.log}"

export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
command -v tmux >/dev/null 2>&1 || sudo apt-get install -y -qq tmux

VENV="$HOME/.venvs/vllm-tpu"
if ! "$VENV/bin/python" -c "import vllm" >/dev/null 2>&1; then
  uv venv --python 3.12 "$VENV"
  uv pip install --python "$VENV/bin/python" "vllm-tpu==0.23.0"
  uv pip install --python "$VENV/bin/python" hf_transfer || true
fi
# Optional extra pins applied AFTER the base install. gemma-4-31B fails to load under
# transformers 5.15.0 with AmbiguousGlobalPerLayerAttributeError ('head_dim' is per-layer),
# while qwen is fine, so the pin is per-model rather than global.
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

# The XLA cache goes on LOCAL disk, never the gcsfuse mount: writing it
# during compile has flaked with "transport endpoint not connected".
export VLLM_XLA_CACHE_PATH="$HOME/vllm-xla-cache-local"
# torchax-on-jax compiles through JAX's compilation cache, which ignores
# VLLM_XLA_CACHE_PATH -- point BOTH at the same dir or the save-back
# uploads an empty tree (measured: job 3744162, 55 min up, zero objects).
export JAX_COMPILATION_CACHE_DIR="$VLLM_XLA_CACHE_PATH"
mkdir -p "$VLLM_XLA_CACHE_PATH"
if [ -n "$XLA_CACHE_GCS" ]; then
  gsutil -m -q rsync -r "$XLA_CACHE_GCS" "$VLLM_XLA_CACHE_PATH" 2>/dev/null \
    && echo "XLA cache restored from ${XLA_CACHE_GCS}" || echo "XLA cache restore failed (will compile)"
  # SAVE-BACK: the restore was one-way, so any config not already cached
  # (e.g. the first 32k-context boot) recompiled EVERY run. Once the engine
  # answers, rsync the compiled cache up -- first boot at a new config pays
  # the compile once, every later boot restores it.
  ( for _ in $(seq 1 360); do
      curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && break
      sleep 20
    done
    if curl -sf "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
      sleep 60   # let warmup compilation finish writing
      # Lazy compile (no precompile) means shapes keep landing through the
      # first request waves -- re-sync a few times so late compiles bank too.
      for _ in 1 2 3; do
        _n=$(find "$VLLM_XLA_CACHE_PATH" -type f | wc -l)
        gsutil -m -q rsync -r "$VLLM_XLA_CACHE_PATH" "$XLA_CACHE_GCS" 2>/dev/null \
          && echo "XLA cache saved back to ${XLA_CACHE_GCS} (${_n} files)"
        sleep 600
      done
    fi ) &
fi

export MODEL_IMPL_TYPE=vllm          # required for the Qwen3.5 arch on TPU
export TPU_BACKEND_TYPE=torchax
export CLOUD_TPU_TASK_ID=0           # standalone single-host server
# SKIP the all-buckets precompile. A cold 22k precompile has been measured at
# ~55 min on this stack, which is a third of the probe's entire wall budget;
# skipping it compiles the shapes we actually use, on first use.
export SKIP_JAX_PRECOMPILE=1
# CHIP SELECTION. On a v5p-8 the host has exactly the 4 chips TP=4 wants, so
# nothing is pinned. On a wider host (v6e-8 = 8 chips) TP=4 must be pinned to
# a contiguous 4, or libtpu builds an 8-chip mesh the server never asked for.
# SERVE_VISIBLE_CHIPS carries that choice from the launcher.
SERVE_TP=${SERVE_TP:-4}
if [ -n "${SERVE_VISIBLE_CHIPS:-}" ]; then
  export TPU_VISIBLE_CHIPS="$SERVE_VISIBLE_CHIPS"
  echo "[serve] pinned to chips ${TPU_VISIBLE_CHIPS} (TP=${SERVE_TP})"
else
  unset TPU_VISIBLE_CHIPS            # every local chip
  echo "[serve] using all local chips (TP=${SERVE_TP})"
fi

# MAKE THIS HOST A STANDALONE TPU RUNTIME. Without these, libtpu on a v5p-16
# tries to form the full TWO-HOST mesh and blocks -- observed as
#   "TPU backend initialization is taking more than 60.0 seconds.
#    Did you run your code on all TPU hosts?"
# with the engine never reaching the serving loop. 1,1,1 processes x 2,2,1
# chips = the 4 chips of THIS host, which is exactly what TP=4 wants, and it
# is what start_colocated_vllm_tinker.sh sets for its single-worker vLLM.
# Chip mesh must match SERVE_TP: 4 chips are 2x2x1, 8 chips are 4x2x1.
export TPU_PROCESS_BOUNDS=1,1,1
case "$SERVE_TP" in
  8) export TPU_CHIPS_PER_PROCESS_BOUNDS=4,2,1 ;;
  2) export TPU_CHIPS_PER_PROCESS_BOUNDS=2,1,1 ;;
  1) export TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 ;;
  *) export TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 ;;
esac
unset TPU_PROCESS_ADDRESSES

pkill -f "vllm serve" >/dev/null 2>&1 || true
pkill -f "VLLM::EngineCore" >/dev/null 2>&1 || true
sleep 3
rm -f "$LOG"

tmux kill-session -t vllm-probe >/dev/null 2>&1 || true
tmux new-session -d -s vllm-probe \
  "MODEL_IMPL_TYPE=vllm TPU_BACKEND_TYPE=torchax SKIP_JAX_PRECOMPILE=1 CLOUD_TPU_TASK_ID=0 \
   TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
   HF_HOME='$HF_HOME' HF_HUB_ENABLE_HF_TRANSFER=1 VLLM_XLA_CACHE_PATH='$VLLM_XLA_CACHE_PATH' \
   '$VENV/bin/vllm' serve '$MODEL' \
     --served-model-name '$MODEL' \
     --host ${BIND_HOST:-127.0.0.1} --port $PORT \
     --tensor-parallel-size ${SERVE_TP} \
     --max-model-len $MAX_MODEL_LEN \
     --max-num-seqs $MAX_NUM_SEQS \
     --max-num-batched-tokens 4096 \
     --download-dir '$HF_HOME/hub' \
     2>&1 | tee -a '$LOG'"

echo "serve_vllm: launched ${MODEL} on port ${PORT} (log ${LOG})"
