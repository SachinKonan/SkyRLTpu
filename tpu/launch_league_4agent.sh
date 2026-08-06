#!/usr/bin/env bash
# Launch the 4-AGENT league (qwenA,qwenB,gemmaA,gemmaB) on w0 of the v5p64c
# slice (in tmux). One shared PUCT tree (topk_children=4 -> one leaf slot per
# member per parent), POOLED cross-CE (all 3 rivals' rollouts compete for the
# 4 import slots per problem), lambda=0.1, v2 pipelined executor, ray grading.
# Per-member batch 8x32 = 1024 rollouts/step total (equal compute vs the
# 2-agent arms). Tree seeded from league1's step-0 snapshot (gs://.../league1c).
# Usage: launch_league_4agent.sh <qwenB_w2_int> <gemmaA_w4_int> <gemmaB_w6_int>
set -uo pipefail
QB_INT=${1:?usage: launch_league_4agent.sh <qwenB_int> <gemmaA_int> <gemmaB_int>}
GA_INT=${2:?missing gemmaA internal ip}
GB_INT=${3:?missing gemmaB internal ip}
EXP=${EXPERIMENT_NAME:-league1c-4agent-erdos}
mkdir -p ~/skyrl-runs

GCS_RUN=gs://sk7524-tinker-tpu-us-east5/skyrl-runs/league1c
for _r in 1 2 3; do
  gsutil -m rsync -r "$GCS_RUN" ~/skyrl-runs/league1c >> ~/restore.log 2>&1 && break
  echo "restore attempt $_r failed; retrying" >> ~/restore.log; sleep 10
done
tmux kill-session -t league-backup 2>/dev/null
tmux new-session -d -s league-backup "while true; do gsutil -m rsync -r -x '.*wandb/.*|.*\.tmp$|.*\.gstmp$' ~/skyrl-runs/league1c $GCS_RUN >> ~/sidecar.log 2>&1; echo sidecar-rc=\$? >> ~/sidecar.log; sleep 300; done"

tmux kill-session -t league 2>/dev/null
tmux new-session -d -s league "cd ~/ttd-client && \
  EXPERIMENT_NAME=$EXP \
  TTD_RUN_DIR=\$HOME/skyrl-runs/league1c \
  TTD_ENV=erdos_min_overlap \
  TTD_ENSEMBLE_MODELS='Qwen/Qwen3.5-27B:qwen3:qwenA,Qwen/Qwen3.5-27B:qwen3:qwenB,google/gemma-4-31B-it:gemma4:gemmaA,google/gemma-4-31B-it:gemma4:gemmaB' \
  TTD_M0_BASE_URL=http://127.0.0.1:8000 \
  TTD_M0_CONTEXT_WINDOW=18432 TTD_M0_TRAIN_MAX_SEQ=18432 TTD_M0_PHASE1_MAX_TOKENS=13824 \
  TTD_M1_BASE_URL=http://$QB_INT:8000 \
  TTD_M1_CONTEXT_WINDOW=18432 TTD_M1_TRAIN_MAX_SEQ=18432 TTD_M1_PHASE1_MAX_TOKENS=13824 \
  TTD_M2_BASE_URL=http://$GA_INT:8000 \
  TTD_M2_CONTEXT_WINDOW=10240 TTD_M2_TRAIN_MAX_SEQ=10240 TTD_M2_PHASE1_MAX_TOKENS=6656 \
  TTD_M3_BASE_URL=http://$GB_INT:8000 \
  TTD_M3_CONTEXT_WINDOW=10240 TTD_M3_TRAIN_MAX_SEQ=10240 TTD_M3_PHASE1_MAX_TOKENS=6656 \
  TTD_QWEN_TWO_PHASE=1 \
  TTD_CROSS_WEIGHT=${TTD_CROSS_WEIGHT:-0.1} TTD_CROSS_MAX_IMPORTS=4 \
  TTD_TOPK_CHILDREN=4 \
  TTD_ADV_ESTIMATOR=mean_baseline \
  TTD_ELITE_SLOTS=2 TTD_REJECT_TRUNCATED=1 \
  TTD_EVAL_BACKEND=${TTD_EVAL_BACKEND:-ray} TTD_RAY_PAYLOAD=${TTD_RAY_PAYLOAD:-1} \
  TTD_LEAGUE_PIPELINE=${TTD_LEAGUE_PIPELINE:-1} \
  RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:6379} NUM_CPUS_PER_TASK=1 \
  GROUPS_PER_BATCH=16 GROUP_SIZE=16 NUM_EPOCHS=15 \
  LEARNING_RATE=4e-5 LORA_RANK=32 KL_PENALTY_COEF=0 TEMPERATURE=1.0 \
  CONTEXT_WINDOW=18432 EVAL_TIMEOUT=1100 SAVE_EVERY=${SAVE_EVERY:-1} \
  WANDB_PROJECT=tpu-tinker-exps \
  third_party/discover/.venv-ttd-discover/bin/python tpu/run_ttd_ensemble.py \
  2>&1 | tee -a ~/skyrl-runs/$EXP.console.log"
sleep 8
tmux has-session -t league 2>/dev/null && echo LEAGUE4-CLIENT-UP || echo LEAGUE4-CLIENT-FAIL
