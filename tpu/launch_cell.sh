#!/usr/bin/env bash
# Launch ONE Stage-A cell on w0 of a v5p-32 (single model, inside tmux).
#
# Stage A is a 2x3 factorial -- {GRPO, TTD-Discover} x {no-reg, KL, restart} --
# run one cell per slice so a preemption costs one experimental condition rather
# than two, and so cells do not share a Ray grading cluster.
#
# Usage: CELL=grpo-n bash ~/ttd-client/tpu/launch_cell.sh
# Everything else has a default; the caller overrides only what the cell varies.
set -uo pipefail
CELL=${CELL:?set CELL, e.g. grpo-n}
EXP=${EXPERIMENT_NAME:-stageA-$CELL}
RUN=${RUN_DIR_NAME:-stageA-$CELL}
GCS_RUN=${GCS_RUN:-gs://sk7524-tinker-tpu-us-east5/skyrl-runs/$RUN}
mkdir -p ~/skyrl-runs/"$RUN"

# ---- cell knobs (see tpu/stage_a_cells.sh for the six combinations) ---------
OBJECTIVE=${TTD_ADV_ESTIMATOR:-mean_baseline}     # mean_baseline | entropic_adaptive_beta
ELITE=${TTD_ELITE_SLOTS:-2}                       # GRPO: 2, TTD-Discover: 0
GPB=${GROUPS_PER_BATCH:-16}                       # batch x group_size: GRPO 16x32, TTD 8x32
GSZ=${GROUP_SIZE:-32}
KLC=${KL_PENALTY_COEF:-0}                         # arm K: 0.1
RESTART=${TTD_RESTART_RATIO:-0}                   # arm R: 100 (0 disables)
STEPS=${NUM_EPOCHS:-15}
# ---- problem (default Erdos; JSSP cells set the frontier_algo knobs) --------
PROB_ENV=${TTD_ENV:-erdos_min_overlap}
PROB_TYPE=${TTD_PROBLEM_TYPE:-}                   # JSSP: 46
FC_MAX_CASES=${TTD_FCALGO_MAX_CASES:-0}
EVALT=${EVAL_TIMEOUT:-1100}                       # JSSP: 180 (C++ compile+run, fc46-proven)
# Drift measurement is free when KL>0 (the pass runs anyway); at KL=0 it costs a
# base-model logprob pass, so measure every N steps rather than never.
KLMEAS=${TTD_KL_MEASURE_EVERY:-1}

# ---- durable run state -----------------------------------------------------
for _r in 1 2 3; do
  gsutil -m rsync -r "$GCS_RUN" ~/skyrl-runs/"$RUN" >> ~/restore.log 2>&1 && break
  echo "restore attempt $_r failed; retrying" >> ~/restore.log; sleep 10
done

cat > ~/sidecar_"$RUN".sh <<'SIDECAR'
#!/bin/bash
while true; do
  gsutil -m rsync -r -x '.*wandb/.*|.*\.tmp$|.*\.gstmp$' \
    "RUNDIRPLACEHOLDER" "GCSRUNPLACEHOLDER" >> "$HOME/sidecar.log" 2>&1
  echo "sidecar-rc=$? $(date -u +%H:%M:%S)" >> "$HOME/sidecar.log"
  sleep 300
done
SIDECAR
sed -i "s|RUNDIRPLACEHOLDER|$HOME/skyrl-runs/$RUN|; s|GCSRUNPLACEHOLDER|$GCS_RUN|" ~/sidecar_"$RUN".sh
chmod +x ~/sidecar_"$RUN".sh
tmux kill-session -t cell-backup 2>/dev/null
tmux new-session -d -s cell-backup "bash ~/sidecar_$RUN.sh"

# ---- re-register durable checkpoints BEFORE the client starts --------------
# Must be here, not only in bring-up: a relaunch invokes THIS script directly.
# Reads the LOCAL jsonl the restore just pulled down, so it does not depend on
# gcloud auth. reregister_states.py is idempotent.
_jsonl=$(ls ~/skyrl-runs/"$RUN"/tinker_log/*/member_qwen/checkpoints.jsonl 2>/dev/null | head -1)
if [ -s "${_jsonl:-}" ]; then
  python3 ~/ttd-client/tpu/reregister_states.py --base-model Qwen/Qwen3.5-27B --jsonl "$_jsonl" 2>&1 | tail -2
else
  echo "reregister: no local checkpoints.jsonl (fresh lineage)"
fi

tmux kill-session -t cell 2>/dev/null
tmux new-session -d -s cell "cd ~/ttd-client && \
  EXPERIMENT_NAME=$EXP \
  TTD_RUN_DIR=\$HOME/skyrl-runs/$RUN \
  TTD_ENV=$PROB_ENV TTD_PROBLEM_TYPE=$PROB_TYPE TTD_FCALGO_MAX_CASES=$FC_MAX_CASES \
  TTD_ENSEMBLE_MODELS='Qwen/Qwen3.5-27B:qwen3:qwen' \
  TTD_ALLOW_SINGLE_MEMBER=1 \
  TTD_M0_BASE_URL=http://127.0.0.1:8000 \
  TTD_M0_CONTEXT_WINDOW=18432 TTD_M0_TRAIN_MAX_SEQ=18432 TTD_M0_PHASE1_MAX_TOKENS=13824 \
  TTD_QWEN_TWO_PHASE=1 TTD_DISABLE_WANDB_TABLES=1 \
  TTD_CROSS_WEIGHT=0 \
  TTD_ADV_ESTIMATOR=$OBJECTIVE \
  TTD_ELITE_SLOTS=$ELITE TTD_REJECT_TRUNCATED=1 \
  TTD_RESTART_RATIO=$RESTART \
  TTD_KL_MEASURE_EVERY=$KLMEAS \
  TTD_RESUME_STRICT=1 \
  TTD_EVAL_BACKEND=ray TTD_RAY_PAYLOAD=1 \
  RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:6379} NUM_CPUS_PER_TASK=1 \
  GROUPS_PER_BATCH=$GPB GROUP_SIZE=$GSZ NUM_EPOCHS=$STEPS \
  LEARNING_RATE=4e-5 LORA_RANK=32 KL_PENALTY_COEF=$KLC TEMPERATURE=1.0 \
  CONTEXT_WINDOW=18432 EVAL_TIMEOUT=$EVALT SAVE_EVERY=1 \
  WANDB_PROJECT=tpu-tinker-exps \
  third_party/discover/.venv-ttd-discover/bin/python tpu/run_ttd_ensemble.py \
  2>&1 | tee -a ~/skyrl-runs/$EXP.console.log"
sleep 8
tmux has-session -t cell 2>/dev/null && echo "CELL-UP $CELL" || echo "CELL-FAIL $CELL"
