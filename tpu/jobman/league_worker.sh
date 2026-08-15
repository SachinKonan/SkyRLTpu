#!/usr/bin/env bash
# jobman command.cmd for a ctrl-rerun LEAGUE arm on a v5p-64 (8 hosts).
# w0 orchestrates: client venv, qwen engines (trainer w0 + vLLM w1-3), gemma
# engines (trainer w4 + vLLM w5-7), and the Ray grading cluster (head w0
# num-cpus=0, workers w1-7). Engine env = the Stage-A proven per-model values
# (qwen fb 512/64 -- the league-era 2048/8 silently OOM'd on Erdos 18432),
# gemma = the league-validated uniform-10240 config.
set -euo pipefail
: "${JOBMAN_WORKER_ID:?}"; : "${JOBMAN_TPU_INTERNAL_IPS:?}"; : "${CELL:?}"
[ "$JOBMAN_WORKER_ID" = "0" ] || { echo "worker $JOBMAN_WORKER_ID: engines are driven from w0"; exit 0; }

export PATH="$HOME/.local/bin:$PATH"
REPO="$HOME/SkyRLTpu-league"
KEY="$HOME/.ssh/jobman_tpu_ed25519"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
INT="$JOBMAN_TPU_INTERNAL_IPS"
W0INT=$(echo "$INT" | cut -d, -f1)
W4INT=$(echo "$INT" | cut -d, -f5)
ln -sfn "$REPO" "$HOME/ttd-client"

# --- client venv (idempotent) ------------------------------------------------
if [ ! -x "$REPO/third_party/discover/.venv-ttd-discover/bin/python" ]; then
  ( cd "$REPO/third_party/discover" && uv sync --extra math --python 3.11 > ~/venv-build.log 2>&1 \
      && ln -sfn .venv .venv-ttd-discover )
  "$REPO/third_party/discover/.venv-ttd-discover/bin/python" -c "import tinker,numpy,wandb" \
    || { echo "client venv build FAILED"; tail -5 ~/venv-build.log; exit 1; }
fi
echo "client venv OK"

# --- NO w0 HF weight staging -------------------------------------------------
# w0 runs the qwen trainer + the client. The trainer loads MaxText/orbax (the
# skyrl-maxtext-ckpts mirror holds qwen3.5-27b and gemma4-31b), and the client
# needs only tokenizer/config -- neither reads safetensors. Staging weights here
# put 71G of gemma-4 on a 97G boot disk and the tinker server died with
# "No space left on device". Only the vLLM hosts need HF weights, and
# start_vllm_tpu.sh restores those on w1-3 / w5-7 via HF_CACHE_GCS.
# Reclaim anything a previous attempt left behind.
for _d in models--google--gemma-4-31B-it models--Qwen--Qwen3.5-27B; do
  [ -d "$HOME/.cache/huggingface/hub/$_d" ] && {
    echo "removing stray w0 weight cache: $_d"; rm -rf "$HOME/.cache/huggingface/hub/$_d"; }
done
df -h / | tail -1

qwen_tinker_healthy()  { curl -fsS -m6 "http://127.0.0.1:8000/api/v1/get_server_capabilities" >/dev/null 2>&1; }
gemma_tinker_healthy() { curl -fsS -m6 "http://$W4INT:8000/api/v1/get_server_capabilities" >/dev/null 2>&1; }
vllm_half_healthy() {  # $1 = comma worker idxs (2-based cut fields)
  local ip
  for ip in $(echo "$INT" | cut -d, -f"$1" | tr ',' ' '); do
    curl -fsS -m6 "http://$ip:8001/v1/models" >/dev/null 2>&1 || return 1
  done
  return 0
}

PIP="maxtext @ git+https://github.com/SachinKonan/maxtext.git@skyrl/qwen35-dense"

bring_up_qwen() {
  env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$INT" TPU_INTERNAL_IPS="$INT" TPU_NAME="ctrlrerun-$CELL" \
    PROJECT=vision-mix ZONE=us-east5-a REMOTE_USER=sk7524_princeton_edu SSH_KEY_FILE="$KEY" \
    TINKER_BACKEND=tunix TRAIN_WORKERS=0 VLLM_WORKERS=1,2,3 VLLM_RAY_EXECUTOR=0 VLLM_CLIENT_SIDE_ROUND_ROBIN=1 \
    MODEL_NAME=Qwen/Qwen3.5-27B TUNIX_MAXTEXT_MODEL_NAME=qwen3.5-27b TUNIX_MAXTEXT_PIP_SPEC="$PIP" \
    TUNIX_MAXTEXT_KWARGS='{"num_vocab_tiling": 64}' \
    TUNIX_MAX_TARGET_LENGTH=22528 TUNIX_TRAIN_TOKEN_BUDGET=73728 TUNIX_FLCE_TILE_SIZE=512 TRAIN_MICRO_BATCH_SIZE=1 \
    TUNIX_UNIFORM_SEQ_LEN=18432 TUNIX_SEQ_BUCKETS="4096,8192,12288,16384,20480" TUNIX_MINIMAL_FB_OUTPUT=1 \
    VLLM_MAX_MODEL_LEN=22528 VLLM_MAX_NUM_SEQS=128 VLLM_XLA_CACHE_PATH=/home/sk7524_princeton_edu/vllm-xla-cache-local \
    VLLM_XLA_CACHE_GCS="gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-22k" \
    HF_CACHE_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache" \
    VLLM_EXTRA_ARGS="--max-num-batched-tokens 8192 --gpu-memory-utilization 0.85" \
    READY_ATTEMPTS=900 SYNC_SKYRL=1 "$@" \
    bash "$REPO/tpu/start_colocated_vllm_tinker.sh" >> ~/qwen-engine.log 2>&1 || true
}

bring_up_gemma() {
  env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$INT" TPU_INTERNAL_IPS="$INT" TPU_NAME="ctrlrerun-$CELL" \
    PROJECT=vision-mix ZONE=us-east5-a REMOTE_USER=sk7524_princeton_edu SSH_KEY_FILE="$KEY" \
    TINKER_BACKEND=tunix TRAIN_WORKERS=4 VLLM_WORKERS=5,6,7 VLLM_RAY_EXECUTOR=0 VLLM_CLIENT_SIDE_ROUND_ROBIN=1 \
    MODEL_NAME=google/gemma-4-31B-it TUNIX_MAXTEXT_MODEL_NAME=gemma4-31b TUNIX_MAXTEXT_PIP_SPEC="$PIP" \
    TUNIX_MAXTEXT_KWARGS='{"num_vocab_tiling": 32}' \
    TUNIX_MAX_TARGET_LENGTH=10240 TUNIX_TRAIN_TOKEN_BUDGET=40960 TUNIX_FLCE_TILE_SIZE=1024 TRAIN_MICRO_BATCH_SIZE=1 \
    TUNIX_UNIFORM_SEQ_LEN=10240 TUNIX_SEQ_BUCKETS="4096,8192,12288,16384,20480" TUNIX_MINIMAL_FB_OUTPUT=1 \
    VLLM_MAX_MODEL_LEN=16384 VLLM_MAX_NUM_SEQS=128 VLLM_XLA_CACHE_PATH=/home/sk7524_princeton_edu/vllm-xla-cache-local \
    VLLM_XLA_CACHE_GCS="gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-gemma4-31b-16k" \
    HF_CACHE_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4" \
    VLLM_EXTRA_ARGS="--max-num-batched-tokens 8192 --gpu-memory-utilization 0.85" \
    READY_ATTEMPTS=2000 SYNC_SKYRL=1 "$@" \
    bash "$REPO/tpu/start_colocated_vllm_tinker.sh" >> ~/gemma-engine.log 2>&1 || true
}

# --- qwen half ---------------------------------------------------------------
if qwen_tinker_healthy && vllm_half_healthy 2-4; then
  echo "qwen half healthy -- skipping"
elif ! qwen_tinker_healthy && vllm_half_healthy 2-4; then
  echo "qwen trainer down, vLLM healthy -- surgical tinker-only restart"
  tmux kill-session -t skyrl-tinker 2>/dev/null || true; sleep 3
  bring_up_qwen SYNC_SKYRL=0 START_VLLM=0 START_TINKER=1
  qwen_tinker_healthy || { echo "qwen tinker restart FAILED"; tail -6 ~/qwen-engine.log; exit 1; }
else
  echo "qwen half bring-up (512/64, uniform 18432)..."
  bring_up_qwen START_VLLM=1 START_TINKER=1
  qwen_tinker_healthy || { echo "qwen bring-up FAILED"; tail -8 ~/qwen-engine.log; exit 1; }
fi

# --- gemma half --------------------------------------------------------------
if gemma_tinker_healthy && vllm_half_healthy 6-8; then
  echo "gemma half healthy -- skipping"
elif ! gemma_tinker_healthy && vllm_half_healthy 6-8; then
  echo "gemma trainer down, vLLM healthy -- surgical tinker-only restart (on w4)"
  timeout 60 ssh $SSHO sk7524_princeton_edu@"$W4INT" "tmux kill-session -t skyrl-tinker 2>/dev/null" || true; sleep 3
  bring_up_gemma SYNC_SKYRL=0 START_VLLM=0 START_TINKER=1
  gemma_tinker_healthy || { echo "gemma tinker restart FAILED"; tail -6 ~/gemma-engine.log; exit 1; }
else
  echo "gemma half bring-up (1024/32, uniform 10240)..."
  bring_up_gemma START_VLLM=1 START_TINKER=1
  gemma_tinker_healthy || { echo "gemma bring-up FAILED"; tail -8 ~/gemma-engine.log; exit 1; }
fi
echo "both engine halves UP"

# --- ray grading cluster (head w0 num-cpus=0, workers w1-7) ------------------
RAYBIN="$REPO/third_party/discover/.venv-ttd-discover/bin/ray"
if ! "$RAYBIN" status >/dev/null 2>&1; then
  pkill -f "[r]ay/core" 2>/dev/null || true; sleep 2
  "$RAYBIN" start --head --port=6379 --num-cpus=0 --disable-usage-stats >/tmp/ray-head.log 2>&1
  echo "ray head started"
fi
RAYV=$("$REPO/third_party/discover/.venv-ttd-discover/bin/python" -c "import ray; print(ray.__version__)")
for ip in $(echo "$INT" | cut -d, -f2-8 | tr ',' ' '); do
  timeout 900 ssh $SSHO sk7524_princeton_edu@"$ip" "
    export PATH=\$HOME/.local/bin:\$PATH
    pgrep -f '[r]ay/core' >/dev/null && { echo \"ray already on \$(hostname)\"; exit 0; }
    [ -x ~/.venvs/grader/bin/ray ] || {
      uv venv ~/.venvs/grader --python 3.11 >/dev/null 2>&1
      uv pip install --python ~/.venvs/grader/bin/python 'ray==$RAYV' numpy scipy shapely numba scikit-learn psutil >/dev/null 2>&1
    }
    ~/.venvs/grader/bin/ray start --address=$W0INT:6379 --num-cpus=150 --disable-usage-stats >/tmp/ray-worker.log 2>&1 && echo \"ray worker \$(hostname)\"
  " 2>/dev/null || echo "ray worker $ip FAILED (grading degrades, not fatal)"
done
echo "league worker 0 ready ($CELL)"
