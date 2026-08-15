#!/usr/bin/env bash
# jobman command.cmd for a Stage-A cell (runs on ALL workers, branches on
# JOBMAN_WORKER_ID). w0 orchestrates: client venv, engines (tinker trainer on
# itself + vLLM on w1-3, via the PROVEN start_colocated_vllm_tinker.sh invoked
# from w0 over the slice's internal IPs with the jobman ssh key), and the Ray
# grading cluster. Other workers no-op -- w0 reaches them over ssh exactly the
# way the login-node bring-up did, so the engine recipe stays byte-identical.
#
# Idempotent: healthy engines are detected and left alone, so a jobman loop
# iteration after a mere monitor-ssh hiccup does not restart a working slice.
set -euo pipefail
: "${JOBMAN_WORKER_ID:?}"; : "${JOBMAN_TPU_INTERNAL_IPS:?}"; : "${CELL:?}"
[ "$JOBMAN_WORKER_ID" = "0" ] || { echo "worker $JOBMAN_WORKER_ID: engines are driven from w0"; exit 0; }

export PATH="$HOME/.local/bin:$PATH"
REPO="$HOME/SkyRLTpu-league"
KEY="$HOME/.ssh/jobman_tpu_ed25519"
SSHO="-i $KEY -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
INT="$JOBMAN_TPU_INTERNAL_IPS"
W0INT=$(echo "$INT" | cut -d, -f1)
ln -sfn "$REPO" "$HOME/ttd-client"

# --- client venv (idempotent) ------------------------------------------------
if [ ! -x "$REPO/third_party/discover/.venv-ttd-discover/bin/python" ]; then
  ( cd "$REPO/third_party/discover" && uv sync --extra math --python 3.11 > ~/venv-build.log 2>&1 \
      && ln -sfn .venv .venv-ttd-discover )
  "$REPO/third_party/discover/.venv-ttd-discover/bin/python" -c "import tinker,numpy,wandb" \
    || { echo "client venv build FAILED"; tail -5 ~/venv-build.log; exit 1; }
fi
echo "client venv OK"

# --- engines: skip when healthy ---------------------------------------------
engines_healthy() {
  curl -fsS -m6 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1 || return 1
  local ip
  for ip in $(echo "$INT" | cut -d, -f2-4 | tr ',' ' '); do
    curl -fsS -m6 "http://$ip:8001/v1/models" >/dev/null 2>&1 || return 1
  done
  return 0
}
tinker_healthy() { curl -fsS -m6 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1; }
vllm_healthy() {
  local ip
  for ip in $(echo "$INT" | cut -d, -f2-4 | tr ',' ' '); do
    curl -fsS -m6 "http://$ip:8001/v1/models" >/dev/null 2>&1 || return 1
  done
  return 0
}
# --- model dimension (cell prefix g- = gemma-4-31B, else qwen3.5-27B) --------
# Gemma values are the league-validated uniform-10240 config (bringup_v5p64
# step 4b): tile 1024 / nvt 32 / budget 40960 / vLLM 16k with its own caches.
# Qwen cells keep the Stage-A per-suffix tile logic below, byte-identical.
PIP="maxtext @ git+https://github.com/SachinKonan/maxtext.git@skyrl/qwen35-dense"
VLLM_IMPL=vllm
TPUINF_REF=skyrl/v0.23.0-lora
VLLM_XARGS="--max-num-batched-tokens 8192 --gpu-memory-utilization 0.85"
HF_OFFLINE=0
case "$CELL" in
  g-*)
    MODEL_NAME=google/gemma-4-31B-it; MAXTEXT_MODEL=gemma4-31b
    MAXTGT=10240; BUDGET=40960; UNIFORM=10240
    VLLM_LEN=16384
    XLA_GCS="gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-gemma4-31b-16k"
    HF_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache-gemma4"
    ;;
  m-*)
    # Muse-Glimmer-30B: MAXTEXT.md training recipe (fork @4f65ba509, remat full,
    # FSDP-4, FLCE 2048) + rs-study serving shape (flax_nnx impl, TP=4, 16384,
    # page 16 above 8k ctx). Serving model code lives on the muse branch of the
    # tpu-inference fork (contains skyrl/v0.23.0-lora). Orbax pre-seeded at
    # skyrl-maxtext-ckpts/muse-glimmer-30b; HF weights in the shared hf-cache.
    MODEL_NAME=meta-models/Muse-Glimmer-30B; MAXTEXT_MODEL=muse-glimmer-30b
    PIP="maxtext @ git+https://github.com/SachinKonan/maxtext.git@4f65ba509"
    MAXTGT=24576; BUDGET=65536; UNIFORM=16384
    VLLM_LEN=16384
    VLLM_IMPL=flax_nnx
    TPUINF_REF=42e4d796f3da210c07218a92a26a9c2db840bb94
    VLLM_XARGS="--max-num-batched-tokens 8192 --gpu-memory-utilization 0.85 --block-size 16"
    # Dedicated per-shape prefix (16384/TP4/b128/block16), like qwen's -22k and
    # gemma's -16k. Starts empty: first muse node compiles cold (~40min) and the
    # seed-back publishes it; the shared flat pool has NO muse entries (its
    # newest objects predate the port) and restoring 6G of foreign entries is waste.
    XLA_GCS="gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-mg-16k"
    HF_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache"
    # Hub repo is public (verified gated:false); cache is pre-staged for speed
    # and the hub stays available as fallback, so offline mode is NOT forced.
    HF_OFFLINE=0
    ;;
  *)
    MODEL_NAME=Qwen/Qwen3.5-27B; MAXTEXT_MODEL=qwen3.5-27b
    MAXTGT=22528; BUDGET=73728; UNIFORM=18432
    VLLM_LEN=22528
    XLA_GCS="gs://sk7524-tinker-tpu-us-east5/vllm-xla-cache-22k"
    HF_GCS="gs://sk7524-tinker-tpu-us-east5/hf-cache"
    ;;
esac
pick_tiles() {
  case "$MAXTEXT_MODEL" in
    gemma4-31b)
      FLCE_TILE=1024; VOCAB_TILING=32
      MT_KWARGS="{\"num_vocab_tiling\": $VOCAB_TILING}" ;;
    muse-glimmer-30b)
      FLCE_TILE=2048; VOCAB_TILING=32
      MT_KWARGS="{\"remat_policy\": \"full\", \"ici_fsdp_parallelism\": 4, \"num_vocab_tiling\": $VOCAB_TILING}" ;;
    *)
      case "$CELL" in
        *k-j) FLCE_TILE=512; VOCAB_TILING=64 ;;
        *-j)  FLCE_TILE=2048; VOCAB_TILING=8 ;;
        *)    FLCE_TILE=512; VOCAB_TILING=64 ;;
      esac
      MT_KWARGS="{\"num_vocab_tiling\": $VOCAB_TILING}" ;;
  esac
  case "$CELL" in
    grpo-k|ttd-k) SCORE_FIXED=18432 ;;
    *)            SCORE_FIXED=0 ;;
  esac
}

# w0-local HF cache for the trainer + client (scoped to this model's dir).
HF_DIR="models--${MODEL_NAME//\//--}"
mkdir -p "$HOME/.cache/huggingface/hub/$HF_DIR"
gsutil -m -q rsync -r "$HF_GCS/$HF_DIR" "$HOME/.cache/huggingface/hub/$HF_DIR" 2>/dev/null \
  && echo "w0 HF cache restored ($HF_DIR)" || echo "w0 HF cache restore failed (hub fallback for public models)"

if engines_healthy; then
  echo "engines already healthy -- skipping bring-up"
elif ! tinker_healthy && vllm_healthy; then
  # Surgical recovery for a wedged/dead TRAINER with healthy samplers (the
  # ENGINE-SICK case): restart only the tinker server -- a process restart is
  # the only defragmentation the TPU runtime has, and bouncing three healthy
  # vLLM workers would waste ~20 min of cache reloads for nothing. The client
  # re-registers against the fresh registry at its next launch.
  echo "trainer down, vLLM healthy -- surgical tinker-only restart"
  tmux kill-session -t skyrl-tinker 2>/dev/null || true; sleep 3
  pick_tiles
  env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$INT" TPU_INTERNAL_IPS="$INT" TPU_NAME="stagea-$CELL" \
    PROJECT=vision-mix ZONE=us-east5-a REMOTE_USER=sk7524_princeton_edu SSH_KEY_FILE="$KEY" \
    TINKER_BACKEND=tunix TRAIN_WORKERS=0 VLLM_WORKERS=1,2,3 VLLM_RAY_EXECUTOR=0 VLLM_CLIENT_SIDE_ROUND_ROBIN=1 \
    VLLM_MODEL_IMPL_TYPE="$VLLM_IMPL" TPU_INFERENCE_FORK_REF="$TPUINF_REF" HF_HUB_OFFLINE="$HF_OFFLINE" \
    MODEL_NAME="$MODEL_NAME" TUNIX_MAXTEXT_MODEL_NAME="$MAXTEXT_MODEL" TUNIX_MAXTEXT_PIP_SPEC="$PIP" \
    TUNIX_MAXTEXT_KWARGS="$MT_KWARGS" \
    TUNIX_MAX_TARGET_LENGTH=$MAXTGT TUNIX_TRAIN_TOKEN_BUDGET=$BUDGET TUNIX_FLCE_TILE_SIZE=$FLCE_TILE TRAIN_MICRO_BATCH_SIZE=1 \
    TUNIX_UNIFORM_SEQ_LEN=$UNIFORM TUNIX_SEQ_BUCKETS="4096,8192,12288,16384,20480" TUNIX_MINIMAL_FB_OUTPUT=1 \
    SKYRL_SCORE_FIXED_LEN=$SCORE_FIXED \
    READY_ATTEMPTS=900 SYNC_SKYRL=0 START_VLLM=0 START_TINKER=1 \
    bash "$REPO/tpu/start_colocated_vllm_tinker.sh" > ~/tinker-restart.log 2>&1 || true
  if ! tinker_healthy; then
    # ESCALATE, do not just exit: the next attempt would see vLLM still healthy,
    # take this same surgical branch, and fail identically -- an infinite loop
    # (observed live: 96 retries on g-ttd-n-j, ~15h and six slices burned, zero
    # rows). Tear down vLLM so the next attempt is forced through a FULL
    # bring-up, which is the only path that rebuilds trainer state from scratch.
    echo "tinker-only restart FAILED -- tearing down vLLM to force a full rebuild next attempt"
    tail -6 ~/tinker-restart.log 2>/dev/null || true
    for ip in $(echo "$INT" | cut -d, -f2-4 | tr ',' ' '); do
      timeout 60 ssh $SSHO sk7524_princeton_edu@"$ip" \
        "tmux kill-session -t skyrl-vllm 2>/dev/null; pkill -f '[v]llm serve' 2>/dev/null; true" 2>/dev/null || true
    done
    exit 1
  fi
  echo "trainer restarted (vLLM untouched)"
else
  echo "engine bring-up ($MAXTEXT_MODEL uniform=$UNIFORM budget=$BUDGET)..."
  # Erdos cells (long sequences, 18432-class fb buckets) OOM'd at compile with
  # the league tiles on these builds: HLO temporaries 111G vs 95.7G/chip, every
  # train step, silently caught by the ensemble guard -- the cells sampled
  # without training. Heavier tiling (the values the gemma engine has always
  # used) shrinks the fb program. JSSP sequences land in small buckets and
  # trained fine, so -j cells keep the faster original tiles.
  # K arms get small tiles even on JSSP: the penalty pass pins its own ~16G
  # scoring arena beside the fb arena, and grpo-k-j proved 2048/8 + penalty
  # does not fit (1/9 steps trained). Non-K JSSP keeps the faster tiles.
  pick_tiles
  env TPU_SSH_MODE=direct TPU_EXTERNAL_IPS="$INT" TPU_INTERNAL_IPS="$INT" TPU_NAME="stagea-$CELL" \
    PROJECT=vision-mix ZONE=us-east5-a REMOTE_USER=sk7524_princeton_edu SSH_KEY_FILE="$KEY" \
    TINKER_BACKEND=tunix TRAIN_WORKERS=0 VLLM_WORKERS=1,2,3 VLLM_RAY_EXECUTOR=0 VLLM_CLIENT_SIDE_ROUND_ROBIN=1 \
    VLLM_MODEL_IMPL_TYPE="$VLLM_IMPL" TPU_INFERENCE_FORK_REF="$TPUINF_REF" HF_HUB_OFFLINE="$HF_OFFLINE" \
    MODEL_NAME="$MODEL_NAME" TUNIX_MAXTEXT_MODEL_NAME="$MAXTEXT_MODEL" TUNIX_MAXTEXT_PIP_SPEC="$PIP" \
    TUNIX_MAXTEXT_KWARGS="$MT_KWARGS" \
    TUNIX_MAX_TARGET_LENGTH=$MAXTGT TUNIX_TRAIN_TOKEN_BUDGET=$BUDGET TUNIX_FLCE_TILE_SIZE=$FLCE_TILE TRAIN_MICRO_BATCH_SIZE=1 \
    TUNIX_UNIFORM_SEQ_LEN=$UNIFORM TUNIX_SEQ_BUCKETS="4096,8192,12288,16384,20480" TUNIX_MINIMAL_FB_OUTPUT=1 \
    SKYRL_SCORE_FIXED_LEN=$SCORE_FIXED \
    VLLM_MAX_MODEL_LEN=$VLLM_LEN VLLM_MAX_NUM_SEQS=128 VLLM_XLA_CACHE_PATH=/home/sk7524_princeton_edu/vllm-xla-cache-local \
    VLLM_XLA_CACHE_GCS="$XLA_GCS" \
    HF_CACHE_GCS="$HF_GCS" \
    VLLM_EXTRA_ARGS="$VLLM_XARGS" \
    READY_ATTEMPTS=900 SYNC_SKYRL=1 START_VLLM=1 START_TINKER=1 \
    bash "$REPO/tpu/start_colocated_vllm_tinker.sh" > ~/engine-bringup.log 2>&1 || true
  curl -fsS -m8 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null 2>&1 \
    || { echo "engine bring-up FAILED"; tail -8 ~/engine-bringup.log; exit 1; }
  echo "engines UP"
fi

# --- ray grading cluster (idempotent) ---------------------------------------
RAYBIN="$REPO/third_party/discover/.venv-ttd-discover/bin/ray"
if ! "$RAYBIN" status >/dev/null 2>&1; then
  pkill -f "ray/core" 2>/dev/null || true; sleep 2
  "$RAYBIN" start --head --port=6379 --num-cpus=0 --disable-usage-stats >/tmp/ray-head.log 2>&1
  echo "ray head started"
fi
RAYV=$("$REPO/third_party/discover/.venv-ttd-discover/bin/python" -c "import ray; print(ray.__version__)")
for ip in $(echo "$INT" | cut -d, -f2-4 | tr ',' ' '); do
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
echo "cell worker 0 ready ($CELL)"
