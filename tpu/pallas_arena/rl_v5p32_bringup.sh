#!/usr/bin/env bash
# Bring up the pallas-arena RL cell on ONE v5p-32 (the agreed topology):
#   worker 0    trainer  (tinker API + MaxText fb, sweep1 sizing)
#   workers 1,2 vLLM     (Qwen3.5-27B, TP=8 each, client-side round robin)
#   TP counts TensorCores: a v5p host exposes 8 cores; TP=4 made vLLM split
#   them TP=4 x DP=2, and DP>=2 hard-rejects LoRA (needed for weight sync).
#   worker 3    judge    (arena queue :8791 + rg_lru judge worker, local poll)
#
# Run from the login node AFTER the jobman slice
# (tpu/runs/yamls/skyrl_qwen35arena_v5p32_spot.yaml) is READY:
#   bash tpu/pallas_arena/rl_v5p32_bringup.sh
#
# Engine sizing is sweep1's proven config verbatim: UNIFORM=18432
# BUDGET=73728 mb=1 (the [4,20480] shape was MEASURED to fail -- fb arena
# scales ~quadratically; 18432 leaves ~15G margin). Serving 18432/seqs 64.
set -uo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"

TPU_NAME="${TPU_NAME:-sk7524-qwen35arena-v5p32-east5a_spot}"
PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
REMOTE_USER="${REMOTE_USER:-sk7524_princeton_edu}"
SSH_KEY_FILE="${SSH_KEY_FILE:-$HOME/.ssh/google_compute_engine}"
QUEUE_PORT="${QUEUE_PORT:-8791}"
CACHE="${CACHE:-gs://sk7524-pallas-arena-us-east5/reward-cache-rl-rglru-v1}"
COMPILE_BUDGET_S="${COMPILE_BUDGET_S:-90}"

tssh() { # tssh <worker> <cmd>
  timeout 900 gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
    --project="$PROJECT" --worker="$1" --command="$2"
}

echo "=== [A] trainer (w0) + 2 vLLM engines (w1,w2) $(date +%H:%M:%S) ==="
env TPU_NAME="$TPU_NAME" PROJECT="$PROJECT" ZONE="$ZONE" \
  REMOTE_USER="$REMOTE_USER" SSH_KEY_FILE="$SSH_KEY_FILE" \
  TINKER_BACKEND=tunix TRAIN_WORKERS=0 VLLM_WORKERS=1,2 \
  VLLM_CLIENT_SIDE_ROUND_ROBIN=1 VLLM_TP_SIZE=8 VLLM_ENGINES_PER_HOST=1 \
  MODEL_NAME=Qwen/Qwen3.5-27B TUNIX_MAXTEXT_MODEL_NAME=qwen3.5-27b \
  TUNIX_MAXTEXT_PIP_SPEC="maxtext @ git+https://github.com/SachinKonan/maxtext.git@skyrl/qwen35-dense" \
  TUNIX_MAX_TARGET_LENGTH=18432 TUNIX_UNIFORM_SEQ_LEN=18432 TUNIX_TRAIN_TOKEN_BUDGET=73728 \
  TRAIN_MICRO_BATCH_SIZE=1 TUNIX_MINIMAL_FB_OUTPUT=1 \
  VLLM_MAX_MODEL_LEN=18432 VLLM_MAX_NUM_SEQS=64 \
  VLLM_EXTRA_ARGS="--max-num-batched-tokens 4096" \
  SYNC_SKYRL="${SYNC_SKYRL:-1}" \
  bash tpu/start_colocated_vllm_tinker.sh
echo "=== [A] start_colocated rc=$? ==="

echo "=== [B] judge + queue on worker 3 $(date +%H:%M:%S) ==="
# Arena code by tarball (scp -r of a tree over gcloud is slow and flaky).
tar -czf /tmp/arena-code.tar.gz -C tpu pallas_arena
timeout 600 gcloud compute tpus tpu-vm scp /tmp/arena-code.tar.gz \
  "${TPU_NAME}:/tmp/arena-code.tar.gz" --zone="$ZONE" --project="$PROJECT" --worker=3
tssh 3 "mkdir -p ~/arena && tar -xzf /tmp/arena-code.tar.gz -C ~/arena"
tssh 3 "bash ~/arena/pallas_arena/phase2/provision_judge.sh"

# Queue (CPU FastAPI) local to the judge; the client reaches it via ssh -L.
tssh 3 "tmux kill-session -t arena-queue 2>/dev/null; tmux new-session -d -s arena-queue \
  'PYTHONPATH=~/arena ~/arena-venv/bin/python -m pallas_arena.judge.queue \
     --port ${QUEUE_PORT} --host 127.0.0.1 --lease-timeout 240 \
     2>&1 | tee -a ~/queue.log'"

# Judge worker: restart loop in tmux (compile-budget watchdog exits 17 by
# design after posting its terminal verdict; the loop brings it back).
tssh 3 "tmux kill-session -t arena-judge 2>/dev/null; tmux new-session -d -s arena-judge \
  'while true; do cd ~/arena && ARENA_CHILD_JAX_PLATFORMS=cpu ARENA_RLIMIT_GB=32 PYTHONPATH=~/arena \
     ~/arena-venv/bin/python -m pallas_arena.judge.worker \
     --problem rg_lru --queue http://127.0.0.1:${QUEUE_PORT} --sim-mode real \
     --timing-pairs 20 --worker-id w3-judge --compile-budget-s ${COMPILE_BUDGET_S} \
     --cache ${CACHE} --boot-report ~/boot-report.json --poll-s 2 \
     2>&1 | tee -a ~/worker-progress.log; echo \"[loop] worker exited rc=\$?; restarting in 10s\"; sleep 10; done'"

echo "=== [C] health $(date +%H:%M:%S) ==="
tssh 0 "curl -fsS -m8 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null && echo 'tinker: UP' || echo 'tinker: DOWN'"
tssh 3 "for i in \$(seq 1 30); do curl -fsS -m5 http://127.0.0.1:${QUEUE_PORT}/status && break; sleep 5; done; echo"
echo "=== bring-up complete; judge boot (elections + calibration) continues in ~/worker-progress.log on w3 ==="
