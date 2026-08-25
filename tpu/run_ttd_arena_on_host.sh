#!/usr/bin/env bash
# Pallas-arena RL client ON w0 of the cell (the league pattern), inside tmux.
#
# WHY ON-HOST, not slurm: league runs its client on w0 talking to the
# colocated tinker server, and it is strictly better here --
#   * no walltime (a slurm job's clock forces resubmission plumbing),
#   * no ssh tunnels (every -L is a thing that drops mid-compile; the gemma
#     probe's "empty verdicts" were exactly that),
#   * SHARED FATE with the slice: the client dies with the cell and returns
#     with it -- never alive-but-useless against a dead trainer,
#   * the grading queue is reached over the cell's own network (w3:8791),
#     so submission is a plain intra-cell HTTP call.
# The trainer host's ~200 CPU cores are otherwise idle; the client is
# coordination work, and arena grading happens on w3's chips regardless.
#
# Usage (in a tmux on worker 0, after rl_v5p32_bringup.sh):
#   EXPERIMENT_NAME=rglru-arena-grpo QUEUE_HOST=10.202.x.y \
#     bash ~/ttd-client/tpu/run_ttd_arena_on_host.sh
set -uo pipefail

CLIENT_ROOT="${CLIENT_ROOT:-$HOME/ttd-client}"
: "${EXPERIMENT_NAME:?set EXPERIMENT_NAME}"
: "${QUEUE_HOST:?set QUEUE_HOST (internal IP of the judge worker, w3)}"

# --- server + grading queue: both colocated on the slice, no tunnels ---
export TINKER_BASE_URL="${TINKER_BASE_URL:-http://127.0.0.1:8000}"
export TINKER_API_KEY="${TINKER_API_KEY:-tml-dummy}"
export ARENA_QUEUE_URL="${ARENA_QUEUE_URL:-http://${QUEUE_HOST}:${QUEUE_PORT:-8791}}"
export ARENA_WAIT_TIMEOUT="${ARENA_WAIT_TIMEOUT:-3600}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-27B}"
export RENDERER_NAME="${RENDERER_NAME:-qwen3}"

# --- discover knobs: sweep1's proven config, arena env ---
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
export WANDB_MODE="${WANDB_MODE:-online}"

# Grading is REMOTE (w3 judges); the local "eval" path only composes the
# contract and posts to the queue, so it needs few workers and a long wait.
export TTD_EVAL_BACKEND=local
export TTD_SAFE_GRADE_MAX_WORKERS="${TTD_SAFE_GRADE_MAX_WORKERS:-32}"
export NUM_CPUS_PER_TASK="${NUM_CPUS_PER_TASK:-1}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-3600}"
export TTD_DISCOVER_SYNC=0
export SKYRLTPU_ROOT="${SKYRLTPU_ROOT:-$CLIENT_ROOT}"

export TTD_RUN_DIR="${TTD_RUN_DIR:-$HOME/skyrl-runs/${EXPERIMENT_NAME}}"
GCS_BACKUP="${GCS_BACKUP:-$HOME/gcs/skyrl-runs/${EXPERIMENT_NAME}}"
export TTD_SAMPLING_CACHE_GCS="${TTD_SAMPLING_CACHE_GCS:-$GCS_BACKUP/sampling_cache}"
mkdir -p "$TTD_RUN_DIR" "$GCS_BACKUP" "$TTD_SAMPLING_CACHE_GCS" 2>/dev/null

# Spot recreation: restore the run dir if this disk is fresh.
if [ ! -d "$TTD_RUN_DIR/tinker_log" ] && [ -d "$GCS_BACKUP/tinker_log" ]; then
  echo "[supervisor] restoring run dir from GCS backup"
  rsync -a "$GCS_BACKUP/" "$TTD_RUN_DIR/"
fi

backup_loop() {
  while true; do
    sleep 300
    rsync -a --exclude 'wandb' --exclude 'sampling_cache' \
      "$TTD_RUN_DIR/" "$GCS_BACKUP/" 2>/dev/null
  done
}
backup_loop &
BACKUP_PID=$!
trap 'kill "$BACKUP_PID" 2>/dev/null' EXIT

echo "[supervisor] ${EXPERIMENT_NAME}: env=$TTD_ENV shape=${GROUPS_PER_BATCH}x${GROUP_SIZE} \
elite=$TTD_ELITE_SLOTS kl=$KL_PENALTY_COEF queue=$ARENA_QUEUE_URL"

# Both dependencies must answer before the client starts: the trainer API and
# the grading queue. A client that starts against a dead queue burns a whole
# step of rollouts into "arena queue error" failure entries.
for dep in "$TINKER_BASE_URL/api/v1/get_server_capabilities" "$ARENA_QUEUE_URL/status"; do
  for i in $(seq 1 120); do
    curl -fsS -m 5 "$dep" >/dev/null 2>&1 && { echo "[supervisor] up: $dep"; break; }
    [ "$i" = 120 ] && echo "[supervisor] WARNING: never reachable: $dep"
    sleep 30
  done
done

attempt=0
while [ "$attempt" -lt 50 ]; do
  attempt=$((attempt + 1))
  echo "[supervisor] attempt $attempt starting $(date -u)"
  bash "$CLIENT_ROOT/tpu/run_ttd_gptoss20b.sh"
  rc=$?
  rsync -a --exclude 'wandb' --exclude 'sampling_cache' "$TTD_RUN_DIR/" "$GCS_BACKUP/" 2>/dev/null
  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] client exited cleanly (run complete) $(date -u)"
    break
  fi
  echo "[supervisor] client died rc=$rc; waiting for server+queue then resuming"
  for i in $(seq 1 120); do
    curl -fsS -m 5 "$TINKER_BASE_URL/api/v1/get_server_capabilities" >/dev/null 2>&1 \
      && curl -fsS -m 5 "$ARENA_QUEUE_URL/status" >/dev/null 2>&1 && break
    sleep 30
  done
done
echo "[supervisor] done $(date -u)"
