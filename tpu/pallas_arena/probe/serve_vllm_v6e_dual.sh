#!/usr/bin/env bash
# TWO sampling-only vLLM engines on ONE v6e-8 host: TP=4 each, chips 0-3 and
# 4-7, ports 8001/8002. Runs ON the TPU VM. Idempotent.
#
# The arm cell's shape: engine 1 serves the rg_lru arm, engine 2 the splash
# arm, so the two problems generate in parallel against one model. Adapted
# from probe/serve_vllm.sh (the campaign's proven single-engine launcher);
# the deltas are the per-engine TPU_VISIBLE_CHIPS split and 32k context.
set -euo pipefail

MODEL="${MODEL:?MODEL required}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
HF_CACHE_GCS="${HF_CACHE_GCS:-}"
XLA_CACHE_GCS="${XLA_CACHE_GCS:-}"     # keyed MODEL-32k-tp4-v6e by the caller
EXTRA_PIP="${EXTRA_PIP:-}"

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
if [ -n "$EXTRA_PIP" ]; then
  echo "applying extra pins: $EXTRA_PIP"
  uv pip install --python "$VENV/bin/python" $EXTRA_PIP || echo "extra pin failed (continuing)"
fi

export HF_HOME="$HOME/.cache/huggingface"
export HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME/hub"
if [ -n "$HF_CACHE_GCS" ]; then
  gsutil -m -q rsync -r "$HF_CACHE_GCS" "$HF_HOME/hub" 2>/dev/null \
    && echo "HF cache restored" || echo "HF cache restore failed (will download)"
fi

# One XLA cache dir per engine (they compile identical programs, but two
# processes writing one dir race); both restore from and save back to the
# SAME GCS prefix, so the second boot of either engine is warm.
pkill -f "vllm serve" >/dev/null 2>&1 || true
pkill -f "VLLM::EngineCore" >/dev/null 2>&1 || true
sleep 3

launch_engine() { # launch_engine <idx> <chips> <port>
  local idx="$1" chips="$2" port="$3"
  local cache="$HOME/vllm-xla-cache-e${idx}"
  mkdir -p "$cache"
  if [ -n "$XLA_CACHE_GCS" ]; then
    gsutil -m -q rsync -r "$XLA_CACHE_GCS" "$cache" 2>/dev/null \
      && echo "engine $idx: XLA cache restored" || echo "engine $idx: cold compile"
    ( for _ in $(seq 1 360); do
        curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1 && break
        sleep 20
      done
      if curl -sf "http://127.0.0.1:${port}/v1/models" >/dev/null 2>&1; then
        sleep 60
        for _ in 1 2 3; do
          gsutil -m -q rsync -r "$cache" "$XLA_CACHE_GCS" 2>/dev/null \
            && echo "engine $idx: XLA cache saved back"
          sleep 600
        done
      fi ) &
  fi
  tmux kill-session -t "vllm-e${idx}" >/dev/null 2>&1 || true
  rm -f "$HOME/vllm-e${idx}.log"
  # v6e: 1 TensorCore per chip; a 4-chip subset of the 2x4 host is a 2x2
  # block -- same standalone-runtime env the single-engine launcher proved,
  # plus TPU_VISIBLE_CHIPS to split the host between the two engines.
  tmux new-session -d -s "vllm-e${idx}" \
    "MODEL_IMPL_TYPE=vllm TPU_BACKEND_TYPE=torchax SKIP_JAX_PRECOMPILE=1 CLOUD_TPU_TASK_ID=0 \
     TPU_VISIBLE_CHIPS='${chips}' TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1 \
     HF_HOME='$HF_HOME' HF_HUB_ENABLE_HF_TRANSFER=1 \
     VLLM_XLA_CACHE_PATH='$cache' JAX_COMPILATION_CACHE_DIR='$cache' \
     '$VENV/bin/vllm' serve '$MODEL' \
       --served-model-name '$MODEL' \
       --host 127.0.0.1 --port ${port} \
       --tensor-parallel-size 4 \
       --max-model-len $MAX_MODEL_LEN \
       --max-num-seqs $MAX_NUM_SEQS \
       --max-num-batched-tokens 4096 \
       --download-dir '$HF_HOME/hub' \
       2>&1 | tee -a '$HOME/vllm-e${idx}.log'"
  echo "engine $idx launched: ${MODEL} chips=${chips} port=${port} ctx=${MAX_MODEL_LEN}"
}

launch_engine 1 "0,1,2,3" 8001
launch_engine 2 "4,5,6,7" 8002
