#!/usr/bin/env bash
# Launch ONE ctrl-rerun league arm's client on w0 of a v5p-64 (inside tmux).
# Two members (qwen local :8000, gemma at w4 :8000), one shared PUCT tree.
# Arm knobs arrive via env from the jobman config: TTD_CROSS_WEIGHT,
# TTD_RESTART_RATIO, TTD_RESTART_CONSEC, TTD_RESTART_AT_STEP, NUM_EPOCHS.
set -uo pipefail
CELL=${CELL:?set CELL, e.g. L-ctrl-n}
GEMMA_INT=${GEMMA_INT:?internal IP of w4}
EXP=${EXPERIMENT_NAME:-ctrlrerun-$CELL}
RUN=${RUN_DIR_NAME:-ctrlrerun-$CELL}
GCS_RUN=${GCS_RUN:-gs://sk7524-tinker-tpu-us-east5/skyrl-runs/$RUN}
mkdir -p ~/skyrl-runs/"$RUN"

STEPS=${NUM_EPOCHS:-20}
CROSS=${TTD_CROSS_WEIGHT:-0}
RESTART=${TTD_RESTART_RATIO:-0}
RESTART_CONSEC=${TTD_RESTART_CONSEC:-3}
RESTART_AT=${TTD_RESTART_AT_STEP:--1}

# ---- durable run state ------------------------------------------------------
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

# ---- asymmetric-lineage guard ------------------------------------------------
# A client death BETWEEN the two members' first checkpoint exports leaves one
# member with a row and the other with none. Strict resume then refuses forever
# ("members disagree on resume batch: [1, 0]") and the arm retry-loops. When
# NOTHING has been banked (no metrics rows), the safe resolution is a coherent
# cold start: clear the orphan rows so both members begin together. Guarded on
# the metrics file being empty, so it can never discard real progress -- and we
# do NOT cold-start weights onto a warm tree, which would be an unlogged
# restart and would contaminate a no-restart arm.
_ml=$(ls ~/skyrl-runs/"$RUN"/tinker_log/*/metrics.jsonl 2>/dev/null | head -1)
_rows=0; [ -s "${_ml:-}" ] && _rows=$(wc -l < "$_ml")
if [ "$_rows" -eq 0 ]; then
  _nq=$(cat ~/skyrl-runs/"$RUN"/tinker_log/*/member_qwen/checkpoints.jsonl 2>/dev/null | wc -l)
  _ng=$(cat ~/skyrl-runs/"$RUN"/tinker_log/*/member_gemma/checkpoints.jsonl 2>/dev/null | wc -l)
  if [ "$_nq" != "$_ng" ]; then
    echo "asymmetric lineage with 0 banked steps (qwen=$_nq gemma=$_ng) -- clearing both for a coherent cold start"
    for _m in member_qwen member_gemma; do
      for _f in ~/skyrl-runs/"$RUN"/tinker_log/*/"$_m"/checkpoints.jsonl; do
        [ -e "$_f" ] && : > "$_f"
      done
    done
  fi
fi

# ---- re-register BOTH members' durable checkpoints BEFORE the client starts -
for _spec in "member_qwen:Qwen/Qwen3.5-27B" "member_gemma:google/gemma-4-31B-it"; do
  _md=${_spec%%:*}; _bm=${_spec##*:}
  _jsonl=$(ls ~/skyrl-runs/"$RUN"/tinker_log/*/"$_md"/checkpoints.jsonl 2>/dev/null | head -1)
  if [ -s "${_jsonl:-}" ]; then
    python3 ~/ttd-client/tpu/reregister_states.py --base-model "$_bm" --jsonl "$_jsonl" 2>&1 | tail -2
  else
    echo "reregister $_md: no local checkpoints.jsonl (fresh lineage)"
  fi
done

tmux kill-session -t cell 2>/dev/null
tmux new-session -d -s cell "cd ~/ttd-client && \
  EXPERIMENT_NAME=$EXP \
  TTD_RUN_DIR=\$HOME/skyrl-runs/$RUN \
  TTD_ENV=erdos_min_overlap \
  TTD_ENSEMBLE_MODELS='Qwen/Qwen3.5-27B:qwen3:qwen,google/gemma-4-31B-it:gemma4:gemma' \
  TTD_M0_BASE_URL=http://127.0.0.1:8000 \
  TTD_M0_CONTEXT_WINDOW=18432 TTD_M0_TRAIN_MAX_SEQ=18432 TTD_M0_PHASE1_MAX_TOKENS=13824 \
  TTD_M1_BASE_URL=http://$GEMMA_INT:8000 \
  TTD_M1_CONTEXT_WINDOW=10240 TTD_M1_TRAIN_MAX_SEQ=10240 TTD_M1_PHASE1_MAX_TOKENS=6656 \
  TTD_QWEN_TWO_PHASE=1 TTD_DISABLE_WANDB_TABLES=1 \
  TTD_CROSS_WEIGHT=$CROSS TTD_CROSS_MAX_IMPORTS=4 \
  TTD_ADV_ESTIMATOR=mean_baseline \
  TTD_ELITE_SLOTS=2 TTD_REJECT_TRUNCATED=1 \
  TTD_RESTART_RATIO=$RESTART \
  TTD_RESTART_CONSEC=$RESTART_CONSEC \
  TTD_RESTART_AT_STEP=$RESTART_AT \
  TTD_KL_MEASURE_EVERY=0 \
  TTD_RESUME_STRICT=1 \
  TTD_MAX_CONSEC_TRAIN_ERR=1 \
  TTD_SAMPLING_PROGRESS_TIMEOUT=0 \
  TTD_EVAL_BACKEND=ray TTD_RAY_PAYLOAD=1 \
  TTD_LEAGUE_PIPELINE=${TTD_LEAGUE_PIPELINE:-1} \
  RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:6379} NUM_CPUS_PER_TASK=1 \
  GROUPS_PER_BATCH=16 GROUP_SIZE=32 NUM_EPOCHS=$STEPS \
  LEARNING_RATE=4e-5 LORA_RANK=32 KL_PENALTY_COEF=0 TEMPERATURE=1.0 \
  CONTEXT_WINDOW=18432 EVAL_TIMEOUT=1100 SAVE_EVERY=1 \
  WANDB_PROJECT=tpu-tinker-exps \
  third_party/discover/.venv-ttd-discover/bin/python tpu/run_ttd_ensemble.py \
  2>&1 | tee -a ~/skyrl-runs/$EXP.console.log"
sleep 8
tmux has-session -t cell 2>/dev/null && echo "CELL-UP $CELL" || echo "CELL-FAIL $CELL"
