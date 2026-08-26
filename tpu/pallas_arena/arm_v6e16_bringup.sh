#!/usr/bin/env bash
# ARM CELL on ONE v6e-16 (2 hosts x 8 chips), one cell PER MODEL:
#
#   w0  SERVING + CLIENT   two vLLM engines, TP=4 each (chips 0-3 / 4-7),
#                          32k context; the arms client runs here in tmux
#                          (league pattern: no slurm, no tunnels, shared fate)
#   w1  GRADING            arena queue (0.0.0.0) + per-test Ray pool over its
#                          8 chips, BOTH problems, P1+tp4 cases, max tp 4
#
# The two problems run in parallel: the rg_lru arm generates against engine
# :8001 while the splash arm generates against :8002, and both submit their
# programs to w1's queue for REAL-silicon verdicts (fwd+bwd, tp4 included).
#
# Usage (login node, after the QR is ACTIVE):
#   MODEL=qwen  TPU_NAME=sk7524-arm16-qwen-v6e16  bash tpu/pallas_arena/arm_v6e16_bringup.sh
#   MODEL=gemma TPU_NAME=sk7524-arm16-gemma-v6e16 bash tpu/pallas_arena/arm_v6e16_bringup.sh
set -uo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"

MODEL="${MODEL:?MODEL required: qwen | gemma}"
TPU_NAME="${TPU_NAME:?TPU_NAME required}"
PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-b}"
QUEUE_PORT="${QUEUE_PORT:-8791}"
COMPILE_BUDGET_S="${COMPILE_BUDGET_S:-90}"
BUCKET=gs://sk7524-pallas-arena-us-east5
SERVE_BUCKET=gs://sk7524-tinker-tpu-us-east5

case "$MODEL" in
  qwen)
    HF_MODEL="Qwen/Qwen3.5-27B"
    HF_GCS="${SERVE_BUCKET}/hf-cache"
    XLA_GCS="${SERVE_BUCKET}/vllm-xla-cache-qwen35-32k-tp4-v6e"
    SERVE_EXTRA_PIP="" ;;
  gemma)
    HF_MODEL="google/gemma-4-31B-it"
    HF_GCS="${SERVE_BUCKET}/hf-cache-gemma4"
    # 32k on v6e TP=4 is a NEW config -- fresh cache prefix, first boot
    # compiles cold and saves back (serve script handles both directions).
    XLA_GCS="${SERVE_BUCKET}/vllm-xla-cache-gemma4-31b-32k-tp4-v6e"
    SERVE_EXTRA_PIP="transformers==5.14.0" ;;
  *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac
CACHE="${CACHE:-${BUCKET}/reward-cache-arm16-${MODEL}-v1}"
JAX_CACHE_GCS="${JAX_CACHE_GCS:-${BUCKET}/judge-jax-cache-v6e-8}"

tssh() { # tssh <worker> <cmd>
  timeout 900 gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE" \
    --project="$PROJECT" --worker="$1" --command="$2"
}
tscp() { # tscp <src> <worker> <dst>
  timeout 600 gcloud compute tpus tpu-vm scp "$1" "${TPU_NAME}:$3" \
    --zone="$ZONE" --project="$PROJECT" --worker="$2"
}

w1_ip=$(timeout 120 gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$ZONE" \
          --project="$PROJECT" --format="value(networkEndpoints[1].ipAddress)")
[ -n "$w1_ip" ] || { echo "FATAL: cannot resolve w1 internal IP"; exit 1; }
echo "=== arm cell ${TPU_NAME} (${MODEL}): w1(judge)=${w1_ip} $(date +%H:%M:%S) ==="

echo "=== [A] serving: two TP=4 engines on w0 ==="
tscp tpu/pallas_arena/probe/serve_vllm_v6e_dual.sh 0 '~/serve_vllm_v6e_dual.sh'
tssh 0 "MODEL='${HF_MODEL}' MAX_MODEL_LEN=32768 MAX_NUM_SEQS=16 \
  HF_CACHE_GCS='${HF_GCS}' XLA_CACHE_GCS='${XLA_GCS}' \
  ${SERVE_EXTRA_PIP:+EXTRA_PIP='${SERVE_EXTRA_PIP}'} bash ~/serve_vllm_v6e_dual.sh"

echo "=== [B] grading: queue + per-test ray pool on w1 ==="
tar -czf /tmp/arena-code.tar.gz -C tpu pallas_arena
tscp /tmp/arena-code.tar.gz 1 /tmp/arena-code.tar.gz
tssh 1 "mkdir -p ~/arena && tar -xzf /tmp/arena-code.tar.gz -C ~/arena"
tssh 1 "bash ~/arena/pallas_arena/phase2/provision_judge.sh"
tssh 1 "tmux kill-session -t arena-queue 2>/dev/null; tmux new-session -d -s arena-queue \
  'PYTHONPATH=~/arena ~/arena-venv/bin/python -m pallas_arena.judge.queue \
     --port ${QUEUE_PORT} --host 0.0.0.0 --lease-timeout 240 \
     2>&1 | tee -a ~/queue.log'"
tssh 1 "RAY_CHIPS=8 bash ~/arena/pallas_arena/phase2/ray_start_tpu.sh"
tssh 1 "tmux kill-session -t arena-judge 2>/dev/null; tmux new-session -d -s arena-judge \
  'while true; do cd ~/arena && PYTHONPATH=~/arena \
     ~/arena-venv/bin/python -m pallas_arena.judge.ray_pool \
     --queue http://127.0.0.1:${QUEUE_PORT} --problems rg_lru,splash_attention \
     --cases \"rg_lru=probe-4x2048x2560,probe-2x1024x2560,probe-8x512x2560,probe-2x4096x2560,probe-4x2048x1024,probe-holdout-2x1500x2560,tp4-4x2048x2560,tp4-holdout-2x1500x2560\" \
     --cases \"splash_attention=probe-h8-s4096,probe-h4-s2048,probe-h16-s1024,probe-h8-s4096-d64,mixtral-8x7b-gqa32x8-s4096,mqa-h32kv1-s4096,deepseek2-16b-s1024-d192-dv128,probe-holdout-h4-s2049,mixtral-holdout-gqa32x8-s2049,tp4-h32-s4096,tp4-gqa32x8-s4096,tp4-mqa-h32kv1-s4096,tp4-holdout-h32-s2049\" \
     --chips 8 --max-tp-width 4 \
     --timing-pairs 20 --compile-budget-s ${COMPILE_BUDGET_S} \
     --cache ${CACHE} --jax-cache-gcs ${JAX_CACHE_GCS} --poll-s 1 \
     2>&1 | tee -a ~/worker-progress.log; echo \"[loop] pool exited rc=\$?; restarting in 10s\"; sleep 10; done'"

echo "=== [C] client tree on w0 ==="
# The arms client needs the probe scripts + seeds; ship the arena tree there too.
tscp /tmp/arena-code.tar.gz 0 /tmp/arena-code.tar.gz
tssh 0 "mkdir -p ~/arena && tar -xzf /tmp/arena-code.tar.gz -C ~/arena; \
  echo 'http://${w1_ip}:${QUEUE_PORT}' > ~/arena-queue-url.txt"

echo "=== [D] health ==="
tssh 1 "for i in \$(seq 1 30); do curl -fsS -m5 http://127.0.0.1:${QUEUE_PORT}/status && break; sleep 5; done; echo"
tssh 0 "curl -fsS -m8 http://${w1_ip}:${QUEUE_PORT}/status >/dev/null && echo 'queue reachable from w0' || echo 'WARNING: queue NOT reachable from w0'"
echo "=== bring-up complete: engines warming on w0 (:8001/:8002, log ~/vllm-e{1,2}.log); judge on w1 ==="
echo "Launch the arms with: tpu/pallas_arena/probe/run_arms_on_cell.sh (on w0, in tmux)"
