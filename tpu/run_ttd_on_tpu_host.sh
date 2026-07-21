#!/usr/bin/env bash
# ttt-discover Erdos client ON a TPU trainer host (worker 0), talking to the
# colocated tinker server at localhost:8000 and grading in-process on the
# host's CPUs (~200 cores; no SLURM, no tunnel, no walltime).
#
# Runs under a supervisor loop: a client crash relaunches it (discover resumes
# weights + PUCT pool from the last SAVE_EVERY checkpoint), and the run dir is
# rsynced to the GCS mount so a spot preemption loses no state that the
# checkpoints don't already cover.
#
# Usage (in a tmux on worker 0):
#   EXPERIMENT_NAME=erdos-qwen35-obj-ttt TTD_ADV_ESTIMATOR=entropic_adaptive_beta \
#     GROUP_SIZE=16 GROUPS_PER_BATCH=32 TTD_ELITE_SLOTS=2 \
#     bash ~/ttd-client/tpu/run_ttd_on_tpu_host.sh
set -uo pipefail

CLIENT_ROOT="${CLIENT_ROOT:-$HOME/ttd-client}"
: "${EXPERIMENT_NAME:?set EXPERIMENT_NAME}"

# --- server: colocated, no tunnel ---
export TINKER_BASE_URL="${TINKER_BASE_URL:-http://127.0.0.1:8000}"
export TINKER_API_KEY="${TINKER_API_KEY:-tml-dummy}"
export MODEL_NAME="${MODEL_NAME:-Qwen/Qwen3.5-27B}"
export RENDERER_NAME="${RENDERER_NAME:-qwen3}"

# --- wandb: shared-mode runs in the TPU experiments project ---
export WANDB_PROJECT="${WANDB_PROJECT:-tpu-tinker-exps}"
export TTD_WANDB_SHARED="${TTD_WANDB_SHARED:-1}"
export WANDB_MODE="${WANDB_MODE:-online}"

# --- discover knobs (defaults = objective-A/B shape; override per arm) ---
export GROUP_SIZE="${GROUP_SIZE:-16}"
export GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-32}"
export NUM_EPOCHS="${NUM_EPOCHS:-30}"
export TTD_ELITE_SLOTS="${TTD_ELITE_SLOTS:-2}"
export SAVE_EVERY="${SAVE_EVERY:-5}"
export CONTEXT_WINDOW="${CONTEXT_WINDOW:-32768}"
# Thinking budget, exact gpt-oss structure: phase 1 = prompt+thinking capped
# at PHASE1_MAX_TOKENS; on budget exhaustion </think> is forced and the
# answer gets the remaining context (QwenTwoPhaseTokenCompleter).
export TTD_QWEN_TWO_PHASE="${TTD_QWEN_TWO_PHASE:-1}"
export PHASE1_MAX_TOKENS="${PHASE1_MAX_TOKENS:-26000}"
# Sequences beyond the trainer fb ceiling are dropped from the gradient
# only (still graded + pooled); see probe_train_len.py.
export TTD_TRAIN_MAX_SEQ="${TTD_TRAIN_MAX_SEQ:-24576}"
export KL_PENALTY_COEF="${KL_PENALTY_COEF:-0.1}"
export TTD_ADV_ESTIMATOR="${TTD_ADV_ESTIMATOR:-entropic_adaptive_beta}"

# --- grading on this host: leave headroom for the api/engine processes ---
CORES_TOTAL="$(nproc)"
export TTD_SAFE_GRADE_MAX_WORKERS="${TTD_SAFE_GRADE_MAX_WORKERS:-$(( CORES_TOTAL > 176 ? 128 : CORES_TOTAL - 48 ))}"
export NUM_CPUS_PER_TASK="${NUM_CPUS_PER_TASK:-1}"
export TTD_EVAL_BACKEND=local
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-1100}"
export TTD_DISCOVER_SYNC=0

export TTD_RUN_DIR="${TTD_RUN_DIR:-$HOME/skyrl-runs/${EXPERIMENT_NAME}}"
GCS_BACKUP="${GCS_BACKUP:-$HOME/gcs/skyrl-runs/${EXPERIMENT_NAME}}"
mkdir -p "$TTD_RUN_DIR" "$GCS_BACKUP"

# Restore from GCS if the local disk is fresh (spot recreation).
if [ ! -d "$TTD_RUN_DIR/tinker_log" ] && [ -d "$GCS_BACKUP/tinker_log" ]; then
  echo "[supervisor] restoring run dir from GCS backup"
  rsync -a "$GCS_BACKUP/" "$TTD_RUN_DIR/"
fi

backup_loop() {
  while true; do
    sleep 300
    rsync -a --exclude 'wandb' "$TTD_RUN_DIR/" "$GCS_BACKUP/" 2>/dev/null
  done
}
backup_loop &
BACKUP_PID=$!
trap 'kill "$BACKUP_PID" 2>/dev/null' EXIT

echo "[supervisor] $EXPERIMENT_NAME: adv=$TTD_ADV_ESTIMATOR shape=${GROUPS_PER_BATCH}x${GROUP_SIZE} elite=$TTD_ELITE_SLOTS kl=$KL_PENALTY_COEF graders=$TTD_SAFE_GRADE_MAX_WORKERS"
attempt=0
while [ "$attempt" -lt 50 ]; do
  attempt=$((attempt + 1))
  echo "[supervisor] attempt $attempt starting $(date -u)"
  bash "$CLIENT_ROOT/tpu/run_ttd_gptoss20b.sh"
  rc=$?
  rsync -a --exclude 'wandb' "$TTD_RUN_DIR/" "$GCS_BACKUP/" 2>/dev/null
  if [ "$rc" -eq 0 ]; then
    echo "[supervisor] client exited cleanly (run complete) $(date -u)"
    break
  fi
  echo "[supervisor] client died rc=$rc; waiting for server then resuming"
  for i in $(seq 1 120); do
    curl -fsS -m 5 "$TINKER_BASE_URL/api/v1/get_server_capabilities" >/dev/null 2>&1 && break
    sleep 30
  done
done
echo "[supervisor] done $(date -u)"
