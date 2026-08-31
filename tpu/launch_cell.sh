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
# Drift measurement (base-model logprob pass on the TRAINER engine). At JSSP's
# short sequences it is nearly free; at Erdos's 18k sequences the pass balloons
# to ~62G HBM, OOMs, and leaves the heap too fragmented for the fb itself --
# this single knob is what made every Erdos cell sample-without-training while
# JSSP trained fine (HBM[prompt_logprobs/oom1] in_use=62.39G). Default ON only
# where it is safe.
case "${TTD_ENV:-erdos_min_overlap}" in
  erdos*) KLMEAS=${TTD_KL_MEASURE_EVERY:-0} ;;
  *)      KLMEAS=${TTD_KL_MEASURE_EVERY:-1} ;;
esac
# ---- model (cell prefix g- = gemma-4-31B, else qwen3.5-27B) -----------------
# Gemma member values are the league-validated ones (launch_league_run.sh M1):
# context/train 10240, phase1 6656, member_gemma lineage dir.
case "$CELL" in
  g-*)
    MODEL_HF=google/gemma-4-31B-it
    MEMBER_SPEC='google/gemma-4-31B-it:gemma4:gemma'
    MEMBER_DIR=member_gemma
    HF_OFFLINE=0
    CTX=10240; PHASE1=6656 ;;
  m-*)
    # Muse-Glimmer-30B: rs-study serving shape (16384); high reasoning strength
    # (template default; xhigh measured null on quality over 960 rollouts).
    MODEL_HF=meta-models/Muse-Glimmer-30B
    MEMBER_SPEC='meta-models/Muse-Glimmer-30B:muse_glimmer_high_reasoning:muse'
    MEMBER_DIR=member_muse
    HF_OFFLINE=0
    CTX=18432; PHASE1=13824 ;;   # qwen-matching: see cell_worker m-* BUDGET comment
  *)
    MODEL_HF=Qwen/Qwen3.5-27B
    MEMBER_SPEC='Qwen/Qwen3.5-27B:qwen3:qwen'
    MEMBER_DIR=member_qwen
    HF_OFFLINE=0
    CTX=18432; PHASE1=13824 ;;
esac
CTX="${CONTEXT_WINDOW:-$CTX}"
PHASE1="${PHASE1_MAX_TOKENS:-$PHASE1}"

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
SESSION=${CELL_SESSION:-cell}
tmux kill-session -t "${SESSION}-backup" 2>/dev/null
tmux new-session -d -s "${SESSION}-backup" "bash ~/sidecar_$RUN.sh"

# ---- seeded-generation guard (META_SEED_ONLY=1) ----------------------------
# A meta generation starts from a seed snapshot the driver staged at step 0.
# The ensemble picks which snapshot to resume by STATE COUNT (resume.py
# pick_resume_snapshot), NOT by step number -- so if a failed launch ever writes
# a cold tree (restore raced, sampler built fresh initial states), that tree
# outgrows the seed after one step and wins the ranking FOREVER. Observed live:
# meta-wt16-fresh-g0-qwen resumed a 78-state cold tree over its 48-state seed
# and ran a whole generation from the wrong starting point.
# While NOTHING has been banked (no metrics rows), the seed is the only
# legitimate snapshot: delete every other one so the ranking cannot pick wrong.
if [ "${META_SEED_ONLY:-0}" = 1 ]; then
  _ml=$(ls ~/skyrl-runs/"$RUN"/tinker_log/*/metrics.jsonl 2>/dev/null | head -1)
  _rows=0; [ -s "${_ml:-}" ] && _rows=$(wc -l < "$_ml")
  if [ "$_rows" -eq 0 ]; then
    for _snap in ~/skyrl-runs/"$RUN"/tinker_log/*/puct_sampler_step_*.json; do
      [ -e "$_snap" ] || continue
      case "$_snap" in
        *puct_sampler_step_000000.json) ;;
        *) echo "seeded start: removing non-seed snapshot $(basename "$_snap")"; rm -f "$_snap" ;;
      esac
    done
    # a cold tree also leaves weight checkpoints; with 0 banked rows they are
    # from the discarded lineage and would resume the run past the seed.
    for _ck in ~/skyrl-runs/"$RUN"/tinker_log/*/member_*/checkpoints.jsonl; do
      [ -e "$_ck" ] && { echo "seeded start: clearing stale $(basename "$(dirname "$_ck")")/checkpoints.jsonl"; : > "$_ck"; }
    done
  fi
fi

# ---- re-register durable checkpoints BEFORE the client starts --------------
# Must be here, not only in bring-up: a relaunch invokes THIS script directly.
# Reads the LOCAL jsonl the restore just pulled down, so it does not depend on
# gcloud auth. reregister_states.py is idempotent.
# REREG_HOST (default local): the tinker registry lives on the member's
# TRAINER host. A stage cell's client runs there, so local is right; a meta
# member's client runs on w0 while its trainer may be w4/w8 -- the register
# must happen THERE (the L-ctrl-x [3, None] crash-loop lesson).
REREG_HOST=${REREG_HOST:-local}
_rereg() {  # $1 = jsonl path
  if [ "$REREG_HOST" = local ]; then
    python3 ~/ttd-client/tpu/reregister_states.py --base-model "$MODEL_HF" --jsonl "$1" 2>&1 | tail -2
  else
    local K="${SSH_KEY_FILE:-$HOME/.ssh/jobman_tpu_ed25519}"
    local U="${REMOTE_USER:-sk7524_princeton_edu}"
    local O="-i $K -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20"
    timeout 60 scp $O ~/ttd-client/tpu/reregister_states.py       "$U"@"$REREG_HOST":~/reregister_states.py >/dev/null 2>&1
    timeout 60 scp $O "$1" "$U"@"$REREG_HOST":~/rr_$(basename "$1") >/dev/null 2>&1
    timeout 90 ssh $O "$U"@"$REREG_HOST"       "python3 ~/reregister_states.py --base-model '$MODEL_HF' --jsonl ~/rr_$(basename "$1")" 2>/dev/null | tail -2
  fi
}
_jsonl=$(ls ~/skyrl-runs/"$RUN"/tinker_log/*/"$MEMBER_DIR"/checkpoints.jsonl 2>/dev/null | head -1)
if [ -s "${_jsonl:-}" ]; then
  _rereg "$_jsonl"
else
  echo "reregister: no local checkpoints.jsonl (fresh lineage)"
fi
# Meta weights-carry: the init state was saved under the PREVIOUS generation's
# run, so its rows are not in this run's jsonl -- register them too.
if [ -s "${EXTRA_REREG_JSONL:-}" ]; then
  _rereg "$EXTRA_REREG_JSONL"
fi

tmux kill-session -t "$SESSION" 2>/dev/null
tmux new-session -d -s "$SESSION" "cd ~/ttd-client && \
  ${EXTRA_TTD_ENV:-} \
  HF_HUB_OFFLINE=$HF_OFFLINE \
  EXPERIMENT_NAME=$EXP \
  TTD_RUN_DIR=\$HOME/skyrl-runs/$RUN \
  TTD_ENV=$PROB_ENV TTD_PROBLEM_TYPE=$PROB_TYPE TTD_FCALGO_MAX_CASES=$FC_MAX_CASES \
  TTD_ENSEMBLE_MODELS=$MEMBER_SPEC \
  TTD_ALLOW_SINGLE_MEMBER=1 \
  TTD_M0_BASE_URL=${TTD_M0_BASE_URL:-http://127.0.0.1:8000} \
  TTD_M0_CONTEXT_WINDOW=$CTX TTD_M0_TRAIN_MAX_SEQ=$CTX TTD_M0_PHASE1_MAX_TOKENS=$PHASE1 \
  TTD_QWEN_TWO_PHASE=1 TTD_DISABLE_WANDB_TABLES=1 \
  TTD_CROSS_WEIGHT=0 \
  TTD_ADV_ESTIMATOR=$OBJECTIVE \
  TTD_ELITE_SLOTS=$ELITE TTD_REJECT_TRUNCATED=1 \
  TTD_RESTART_RATIO=$RESTART \
  TTD_KL_MEASURE_EVERY=$KLMEAS \
  TTD_RESUME_STRICT=1 \
  TTD_SAMPLING_PROGRESS_TIMEOUT=${TTD_SAMPLING_PROGRESS_TIMEOUT:-0} \
  TTD_MAX_CONSEC_TRAIN_ERR=1 \
  TTD_EVAL_BACKEND=ray TTD_RAY_PAYLOAD=1 \
  TTD_LEAGUE_PIPELINE=${TTD_LEAGUE_PIPELINE:-1} \
  RAY_ADDRESS=${RAY_ADDRESS:-127.0.0.1:6379} NUM_CPUS_PER_TASK=1 \
  GROUPS_PER_BATCH=$GPB GROUP_SIZE=$GSZ NUM_EPOCHS=$STEPS \
  LEARNING_RATE=${LEARNING_RATE:-4e-5} LORA_RANK=${LORA_RANK:-32} KL_PENALTY_COEF=$KLC TEMPERATURE=1.0 \
  CONTEXT_WINDOW=$CTX EVAL_TIMEOUT=$EVALT SAVE_EVERY=1 \
  WANDB_PROJECT=tpu-tinker-exps \
  third_party/discover/.venv-ttd-discover/bin/python tpu/run_ttd_ensemble.py \
  2>&1 | tee -a ~/skyrl-runs/$EXP.console.log"
sleep 8
tmux has-session -t "$SESSION" 2>/dev/null && echo "CELL-UP $CELL" || echo "CELL-FAIL $CELL"
