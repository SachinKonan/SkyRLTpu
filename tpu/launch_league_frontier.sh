#!/usr/bin/env bash
# Launch the LAMBDA=0 CONTROL league run on w0 of the v5p64b slice (in tmux).
# Identical to launch_league_run.sh except: cross-imitation OFF (lambda=0),
# run dir/GCS league1d (tree seeded from league1's step-0 snapshot so both
# arms start from the SAME tree), its own experiment name. In-context transfer
# through the shared tree stays ON for its two members -- the A/B isolates the
# weight-space cross-CE term only.
# Usage: bash ~/ttd-client/tpu/launch_league_frontier.sh <gemma_w4_internal_ip>
set -uo pipefail
GEMMA_INT=${1:?usage: launch_league_frontier.sh <gemma_w4_internal_ip>}
EXP=${EXPERIMENT_NAME:-league1d-frontierud-qwen-gemma}
mkdir -p ~/skyrl-runs

GCS_RUN=gs://sk7524-tinker-tpu-us-east5/skyrl-runs/league1d
mkdir -p ~/skyrl-runs/league1d
for _r in 1 2 3; do
  gsutil -m rsync -r "$GCS_RUN" ~/skyrl-runs/league1d >> ~/restore.log 2>&1 && break
  echo "restore attempt $_r failed; retrying" >> ~/restore.log; sleep 10
done
# run-dir backup sidecar. The loop lives in a FILE, not an inline tmux string:
# inline quoting of the rsync -x regex made the session die at spawn on every
# arm (healer had to recreate it every time).
cat > ~/sidecar_league1d.sh <<'SIDECAR'
#!/bin/bash
while true; do
  gsutil -m rsync -r -x '.*wandb/.*|.*\.tmp$|.*\.gstmp$' \
    "$HOME/skyrl-runs/league1d" "GCSRUNPLACEHOLDER" >> "$HOME/sidecar.log" 2>&1
  echo "sidecar-rc=$? $(date -u +%H:%M:%S)" >> "$HOME/sidecar.log"
  sleep 300
done
SIDECAR
sed -i "s|GCSRUNPLACEHOLDER|$GCS_RUN|" ~/sidecar_league1d.sh
chmod +x ~/sidecar_league1d.sh
tmux kill-session -t league-backup 2>/dev/null
tmux new-session -d -s league-backup "bash ~/sidecar_league1d.sh"

tmux kill-session -t league 2>/dev/null
tmux new-session -d -s league "cd ~/ttd-client && \
  EXPERIMENT_NAME=$EXP \
  TTD_RUN_DIR=\$HOME/skyrl-runs/league1d \
  TTD_ENV=frontier_erdos_ud TTD_PROBLEM_TYPE=${TTD_PROBLEM_TYPE:-65536} \
  TTD_ENSEMBLE_MODELS='Qwen/Qwen3.5-27B:qwen3:qwen,google/gemma-4-31B-it:gemma4:gemma' \
  TTD_M0_BASE_URL=http://127.0.0.1:8000 \
  TTD_M0_CONTEXT_WINDOW=18432 TTD_M0_TRAIN_MAX_SEQ=18432 TTD_M0_PHASE1_MAX_TOKENS=13824 \
  TTD_M1_BASE_URL=http://$GEMMA_INT:8000 \
  TTD_M1_CONTEXT_WINDOW=10240 TTD_M1_TRAIN_MAX_SEQ=10240 TTD_M1_PHASE1_MAX_TOKENS=6656 \
  TTD_QWEN_TWO_PHASE=1 \
  TTD_CROSS_WEIGHT=${TTD_CROSS_WEIGHT:-0.1} TTD_CROSS_MAX_IMPORTS=4 \
  TTD_ADV_ESTIMATOR=mean_baseline \
  TTD_ELITE_SLOTS=2 TTD_REJECT_TRUNCATED=1 \
  TTD_EVAL_BACKEND=${TTD_EVAL_BACKEND:-ray} TTD_RAY_PAYLOAD=${TTD_RAY_PAYLOAD:-1} \
  TTD_LEAGUE_PIPELINE=${TTD_LEAGUE_PIPELINE:-1} \
  RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:6379} NUM_CPUS_PER_TASK=1 \
  GROUPS_PER_BATCH=16 GROUP_SIZE=32 NUM_EPOCHS=${NUM_EPOCHS:-40} \
  LEARNING_RATE=4e-5 LORA_RANK=32 KL_PENALTY_COEF=0 TEMPERATURE=1.0 \
  CONTEXT_WINDOW=18432 EVAL_TIMEOUT=1100 SAVE_EVERY=${SAVE_EVERY:-1} \
  WANDB_PROJECT=tpu-tinker-exps \
  third_party/discover/.venv-ttd-discover/bin/python tpu/run_ttd_ensemble.py \
  2>&1 | tee -a ~/skyrl-runs/$EXP.console.log"
sleep 8
tmux has-session -t league 2>/dev/null && echo LEAGUE-D-CLIENT-UP || echo LEAGUE-D-CLIENT-FAIL
