#!/usr/bin/env bash
# Bring up the pallas-arena RL cell on ONE v5p-32 (the agreed topology):
#   worker 0    trainer  (tinker API + MaxText fb, sweep1 sizing)
#   workers 1,2 vLLM     (Qwen3.5-27B, TP=4 each, client-side round robin)
#   TP=4 = one engine per 4-chip v5p host, the standard geometry everywhere
#   (cell_worker.sh: qwen/gemma TP=4 x 1 engine/host; only muse is TP=2 x 2).
#   VLLM_RAY_EXECUTOR=0 is REQUIRED with 2 serving hosts: on the auto/ray
#   path the launcher sets VLLM_DATA_PARALLEL_SIZE=2 across hosts, and DP>=2
#   hard-rejects LoRA (needed for weight sync). Round-robin alone does NOT
#   select the no-ray path.
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
  VLLM_CLIENT_SIDE_ROUND_ROBIN=1 VLLM_RAY_EXECUTOR=0 VLLM_TP_SIZE=4 VLLM_ENGINES_PER_HOST=1 \
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
     --port ${QUEUE_PORT} --host 0.0.0.0 --lease-timeout 240 \
     2>&1 | tee -a ~/queue.log'"

# GRADING EXECUTION. Default is the RAY POOL: the host has 4 chips and a
# single worker uses one, so 3/4 of a host dedicated to grading sits idle
# while grading is what gates every RL step. Ray gives one actor per chip
# (and chip-count-aware dispatch for TP cases later). GRADER=single falls
# back to the one-worker loop, byte-identical to the proven path.
if [ "${GRADER:-ray}" = "ray" ]; then
  tssh 3 "RAY_CHIPS=${RAY_CHIPS:-4} bash ~/arena/pallas_arena/phase2/ray_start_tpu.sh"
  tssh 3 "tmux kill-session -t arena-judge 2>/dev/null; tmux new-session -d -s arena-judge \
    'while true; do cd ~/arena && PYTHONPATH=~/arena JAX_COMPILATION_CACHE_DIR=~/jax-compile-cache \
       ~/arena-venv/bin/python -m pallas_arena.judge.ray_pool \
       --queue http://127.0.0.1:${QUEUE_PORT} --problems ${PROBLEMS:-rg_lru} \
       --actors ${RAY_ACTORS:-4} --width ${RAY_WIDTH:-1} \
       --timing-pairs 20 --compile-budget-s ${COMPILE_BUDGET_S} \
       --cache ${CACHE} --poll-s 1 \
       --jax-cache-gcs ${JAX_CACHE_GCS:-gs://sk7524-pallas-arena-us-east5/judge-jax-cache-v5p-8} \
       2>&1 | tee -a ~/worker-progress.log; echo \"[loop] pool exited rc=\$?; restarting in 10s\"; sleep 10; done'"
else
  tssh 3 "tmux kill-session -t arena-judge 2>/dev/null; tmux new-session -d -s arena-judge \
    'while true; do cd ~/arena && ARENA_CHILD_JAX_PLATFORMS=cpu ARENA_RLIMIT_GB=32 PYTHONPATH=~/arena \
       ~/arena-venv/bin/python -m pallas_arena.judge.worker \
       --problem rg_lru --queue http://127.0.0.1:${QUEUE_PORT} --sim-mode real \
       --timing-pairs 20 --worker-id w3-judge --compile-budget-s ${COMPILE_BUDGET_S} \
       --cache ${CACHE} --boot-report ~/boot-report.json --poll-s 2 \
       2>&1 | tee -a ~/worker-progress.log; echo \"[loop] worker exited rc=\$?; restarting in 10s\"; sleep 10; done'"
fi

echo "=== [C] health $(date +%H:%M:%S) ==="
tssh 0 "curl -fsS -m8 http://127.0.0.1:8000/api/v1/get_server_capabilities >/dev/null && echo 'tinker: UP' || echo 'tinker: DOWN'"
tssh 3 "for i in \$(seq 1 30); do curl -fsS -m5 http://127.0.0.1:${QUEUE_PORT}/status && break; sleep 5; done; echo"
echo "=== [D] RL client ON w0 (league pattern: tmux, no slurm, no tunnels) $(date +%H:%M:%S) ==="
# The client needs the repo (env + discover submodule) on w0. start_colocated
# already rsyncs it when SYNC_SKYRL=1; ~/ttd-client is that tree.
w3_ip=$(timeout 120 gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" --project="$PROJECT" \
          --format="value(networkEndpoints[3].ipAddress)" 2>/dev/null)
if [ -z "$w3_ip" ]; then
  echo "[client] FATAL: cannot resolve w3 internal IP; skipping client launch" >&2
elif [ "${START_CLIENT:-1}" != "1" ]; then
  echo "[client] START_CLIENT=0; skipping"
else
  tssh 0 "test -d ~/ttd-client || ln -sfn ~/SkyRLTpu ~/ttd-client; \
    curl -fsS -m8 http://${w3_ip}:${QUEUE_PORT}/status >/dev/null && echo 'queue reachable from w0' || echo 'WARNING: queue not reachable from w0'"
  tssh 0 "tmux kill-session -t arena-client 2>/dev/null; \
    tmux new-session -d -s arena-client \
    'EXPERIMENT_NAME=${EXPERIMENT_NAME:-rglru-arena-grpo} QUEUE_HOST=${w3_ip} QUEUE_PORT=${QUEUE_PORT} \
     bash ~/ttd-client/tpu/run_ttd_arena_on_host.sh 2>&1 | tee -a ~/arena-client.log'"
  tssh 0 "sleep 5; tmux has-session -t arena-client 2>/dev/null && echo ARENA-CLIENT-UP || echo ARENA-CLIENT-FAIL"
fi

echo "=== bring-up complete; judge boot (elections + calibration) continues in ~/worker-progress.log on w3; client log ~/arena-client.log on w0 ==="
