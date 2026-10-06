#!/usr/bin/env bash
# Launch the league co-training run ON w0 of the v5p-64 (inside tmux).
# Usage: bash ~/ttd-client/tpu/launch_league_run.sh <gemma_w4_internal_ip>
# qwen half is local (w0:8000); gemma half at the w4 internal IP.
set -uo pipefail
GEMMA_INT=${1:?usage: launch_league_run.sh <gemma_w4_internal_ip>}
EXP=${EXPERIMENT_NAME:-league1-qwen-gemma-erdos}
mkdir -p ~/skyrl-runs

# Durable run state (spot slices die ~2h): restore the run dir (PUCT tree,
# metrics, best-seq) from GCS before launching, and keep a backup sidecar
# rsyncing it up every 5 min. With SAVE_EVERY=1 and the re-register step below,
# a preemption is a no-op: tree, LoRA weights, optimizer state and the step
# counter all come back and the run continues where it left off.
GCS_RUN=gs://sk7524-tinker-tpu-us-east5/skyrl-runs/league1
mkdir -p ~/skyrl-runs/league1
for _r in 1 2 3; do
  gsutil -m rsync -r "$GCS_RUN" ~/skyrl-runs/league1 >> ~/restore.log 2>&1 && break
  echo "restore attempt $_r failed; retrying" >> ~/restore.log; sleep 10
done
# run-dir backup sidecar. The loop lives in a FILE, not an inline tmux string:
# inline quoting of the rsync -x regex made the session die at spawn on every
# arm (healer had to recreate it every time).
cat > ~/sidecar_league1.sh <<'SIDECAR'
#!/bin/bash
while true; do
  gsutil -m rsync -r -x '.*wandb/.*|.*\.tmp$|.*\.gstmp$' \
    "$HOME/skyrl-runs/league1" "GCSRUNPLACEHOLDER" >> "$HOME/sidecar.log" 2>&1
  echo "sidecar-rc=$? $(date -u +%H:%M:%S)" >> "$HOME/sidecar.log"
  sleep 300
done
SIDECAR
sed -i "s|GCSRUNPLACEHOLDER|$GCS_RUN|" ~/sidecar_league1.sh
chmod +x ~/sidecar_league1.sh
tmux kill-session -t league-backup 2>/dev/null
tmux new-session -d -s league-backup "bash ~/sidecar_league1.sh"

# Re-register durable checkpoints BEFORE the client starts.
#
# This MUST live here, not only in bring-up step 4z: the guardian relaunches a
# dead client by invoking THIS script directly and never re-runs bring-up, so 4z
# is skipped on every relaunch. That gap silently reset LoRA weights to base on
# all three Erdos arms (2026-08-09) -- the tarballs were on the gcsfuse mount the
# whole time, but the per-VM sqlite registry had no rows, so
# create_training_client_from_state 404'd and each member started from scratch.
#
# Reads the LOCAL checkpoints.jsonl the restore above just pulled down; 4z
# fetched it with gsutil and so died whenever gcloud auth expired.
# reregister_states.py is idempotent (INSERT OR IGNORE, skips absent tarballs).
for _spec in "member_qwen:Qwen/Qwen3.5-27B" "member_gemma:google/gemma-4-31B-it"; do
  _md=${_spec%%:*}; _bm=${_spec##*:}
  _jsonl=$(ls ~/skyrl-runs/league1/tinker_log/*/"$_md"/checkpoints.jsonl 2>/dev/null | head -1)
  if [ -s "${_jsonl:-}" ]; then
    python3 ~/ttd-client/tpu/reregister_states.py --base-model "$_bm" --jsonl "$_jsonl" 2>&1 | tail -2
  else
    echo "reregister $_md: no local checkpoints.jsonl (fresh lineage)"
  fi
done

tmux kill-session -t league 2>/dev/null
tmux new-session -d -s league "cd ~/ttd-client && \
  EXPERIMENT_NAME=$EXP \
  TTD_RUN_DIR=\$HOME/skyrl-runs/league1 \
  TTD_ENV=erdos_min_overlap \
  TTD_ENSEMBLE_MODELS='Qwen/Qwen3.5-27B:qwen3:qwen,google/gemma-4-31B-it:gemma4:gemma' \
  TTD_M0_BASE_URL=http://127.0.0.1:8000 \
  TTD_M0_CONTEXT_WINDOW=18432 TTD_M0_TRAIN_MAX_SEQ=18432 TTD_M0_PHASE1_MAX_TOKENS=13824 \
  TTD_M1_BASE_URL=http://$GEMMA_INT:8000 \
  TTD_M1_CONTEXT_WINDOW=10240 TTD_M1_TRAIN_MAX_SEQ=10240 TTD_M1_PHASE1_MAX_TOKENS=6656 \
  TTD_QWEN_TWO_PHASE=1 TTD_DISABLE_WANDB_TABLES=1 \
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
tmux has-session -t league 2>/dev/null && echo LEAGUE-CLIENT-UP || echo LEAGUE-CLIENT-FAIL
