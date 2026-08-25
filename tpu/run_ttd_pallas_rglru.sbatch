#!/bin/bash
# SUPERSEDED for the v5p-32 cell by tpu/run_ttd_arena_on_host.sh, which runs
# the client ON w0 (league pattern: tmux, no walltime, no ssh tunnels, shared
# fate with the slice). rl_v5p32_bringup.sh launches that automatically.
#
# This slurm-side client is kept for the DETACHED topology: a trainer slice
# plus an independently-scheduled judge fleet, where the client cannot live
# on the cell (e.g. grading a corpus against a serving-only slice).
# ttt-discover RL client: pallas arena RG-LRU on the v5p-32 cell.
#
# The cell (tpu/pallas_arena/rl_v5p32_bringup.sh): w0 trainer, w1-2 vLLM,
# w3 judge+queue. This client tunnels to BOTH w0 (tinker :8000) and w3
# (arena queue :8791) and runs the discover loop with sweep1's proven qwen
# config (CTX 18432 / PHASE1 13824 two-phase / reject-truncated / KL 0 /
# GRPO mean_baseline / elite 2 / lr 4e-5) -- only the env, batch shape and
# eval timeout differ from sweep1:
#   TTD_ENV=pallas_rglru       remote-judged kernel arena (rf3s scaffold)
#   32x2 = 64 rollouts/step    sized to ONE v5p-8 judge host's throughput
#                              (sweep1's 512/step would take ~8h to grade)
#   EVAL_TIMEOUT=3600          queue-depth-dominated wait, not a sandbox cap
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH -t 3-00:00:00
#SBATCH -J ttd-rglru
#SBATCH -o /n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_rglru/slurm_%j.out
#SBATCH -e /n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_rglru/slurm_%j.out
set -euo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
mkdir -p "${REPO}/runs/ttd_rglru"

TPU_NAME="${TPU_NAME:-sk7524-qwen35arena-v5p32-east5a_spot}"
PROJECT="${PROJECT:-vision-mix}"
ZONE="${ZONE:-us-east5-a}"
TPU_REMOTE_USER="${TPU_REMOTE_USER:-sk7524_princeton_edu}"
TPU_SSH_KEY="${TPU_SSH_KEY:-$HOME/.ssh/google_compute_engine}"
TINKER_LOCAL_PORT="${TINKER_LOCAL_PORT:-18050}"
QUEUE_LOCAL_PORT="${QUEUE_LOCAL_PORT:-18791}"
GCLOUD="${GCLOUD:-/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud}"

worker_ip() { # worker_ip <index>
  "$GCLOUD" alpha compute tpus tpu-vm describe "$TPU_NAME" \
    --project="$PROJECT" --zone="$ZONE" \
    --format="value(networkEndpoints[$1].accessConfig.externalIp)" 2>/dev/null
}

# Self-healing tunnels; each re-resolves its worker IP on reconnect (a spot
# recreate changes IPs).
tunnel_loop() { # tunnel_loop <worker> <local_port> <remote_port> <label>
  local w="$1" lp="$2" rp="$3" label="$4" ip
  while true; do
    ip="$(worker_ip "$w" || true)"
    if [[ -z "$ip" ]]; then
      echo "[tunnel:$label] cannot resolve worker $w ip; retry in 60s" >&2; sleep 60; continue
    fi
    echo "[tunnel:$label] 127.0.0.1:${lp} -> ${ip}:${rp}" >&2
    ssh -i "$TPU_SSH_KEY" -o IdentitiesOnly=yes -o ExitOnForwardFailure=yes \
      -o ServerAliveInterval=30 -o ServerAliveCountMax=3 \
      -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=30 \
      -N -L "127.0.0.1:${lp}:127.0.0.1:${rp}" "${TPU_REMOTE_USER}@${ip}" || true
    echo "[tunnel:$label] dropped; reconnect in 15s" >&2
    sleep 15
  done
}
tunnel_loop 0 "$TINKER_LOCAL_PORT" 8000 tinker &
T1=$!
tunnel_loop 3 "$QUEUE_LOCAL_PORT" 8791 queue &
T2=$!
trap 'kill "$T1" "$T2" 2>/dev/null || true' EXIT

for i in $(seq 1 60); do
  ok=0
  curl -fsS -m5 "http://127.0.0.1:${TINKER_LOCAL_PORT}/api/v1/get_server_capabilities" >/dev/null 2>&1 && ok=$((ok+1))
  curl -fsS -m5 "http://127.0.0.1:${QUEUE_LOCAL_PORT}/status" >/dev/null 2>&1 && ok=$((ok+1))
  [[ "$ok" == 2 ]] && { echo "[client] tinker + arena queue reachable"; break; }
  [[ "$i" == 60 ]] && { echo "[client] endpoints unreachable after 5min (tinker+queue=$ok/2)" >&2; exit 1; }
  sleep 5
done

CORES="${SLURM_CPUS_PER_TASK:-32}"
echo "[client] node=$(hostname) job=${SLURM_JOB_ID:-none} cores=${CORES}"

export TINKER_BASE_URL="http://127.0.0.1:${TINKER_LOCAL_PORT}"
export TINKER_API_KEY="${TINKER_API_KEY:-tml-dummy}"
export ARENA_QUEUE_URL="http://127.0.0.1:${QUEUE_LOCAL_PORT}"
export ARENA_WAIT_TIMEOUT="${ARENA_WAIT_TIMEOUT:-3600}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-27B}"
export RENDERER_NAME="${RENDERER_NAME:-qwen3}"

# --- discover knobs: sweep1 verbatim except env/batch/eval ---
export TTD_ENV="${TTD_ENV:-pallas_rglru}"
export GROUP_SIZE="${GROUP_SIZE:-32}"
export GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-2}"
export NUM_EPOCHS="${NUM_EPOCHS:-30}"
export TTD_ELITE_SLOTS="${TTD_ELITE_SLOTS:-2}"
export SAVE_EVERY="${SAVE_EVERY:-5}"
export LEARNING_RATE="${LEARNING_RATE:-4e-5}"
export CONTEXT_WINDOW="${CONTEXT_WINDOW:-18432}"
export PHASE1_MAX_TOKENS="${PHASE1_MAX_TOKENS:-13824}"
export TTD_QWEN_TWO_PHASE="${TTD_QWEN_TWO_PHASE:-1}"
export TTD_REJECT_TRUNCATED="${TTD_REJECT_TRUNCATED:-1}"
export TTD_TRAIN_MAX_SEQ="${TTD_TRAIN_MAX_SEQ:-18432}"
export KL_PENALTY_COEF="${KL_PENALTY_COEF:-0}"
export TTD_ADV_ESTIMATOR="${TTD_ADV_ESTIMATOR:-mean_baseline}"

export WANDB_PROJECT="${WANDB_PROJECT:-tpu-tinker-exps}"
export TTD_WANDB_SHARED="${TTD_WANDB_SHARED:-1}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-rglru-arena-grpo}"

export TTD_EVAL_BACKEND=local
export TTD_SAFE_GRADE_MAX_WORKERS="${TTD_SAFE_GRADE_MAX_WORKERS:-${CORES}}"
export NUM_CPUS_PER_TASK="${NUM_CPUS_PER_TASK:-1}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-3600}"
export TTD_DISCOVER_SYNC="${TTD_DISCOVER_SYNC:-0}"
export SKYRLTPU_ROOT="$REPO"

bash "${REPO}/tpu/run_ttd_gptoss20b.sh"
