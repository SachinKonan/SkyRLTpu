#!/usr/bin/env bash
# Launch one Inkling-2-Small Erdős run on Tinker (client on della-vis1, grading on the gradefarm hosts).
# Usage: launch_run.sh RUN_NAME NUM_STEPS [LORA_SEED]
# Settings mirror the live Erdős merge runs (profile rebut-merge-erdos-muse-a client env) except the Tinker/Inkling
# specifics: model, renderer + effort, client-side thinking cap, lr 1e-5, gradefarm-only seeded 1000 s grading.
set -euo pipefail
NAME=$1; STEPS=$2; LORA_SEED=${3:-1}
WT=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/code/skyrl
DISC=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/code/discover
PY=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/venv/bin/python
RUN=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs/$NAME
REG=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops/registry.json
mkdir -p "$RUN"

# Tinker key from ~/.bashrc (never printed).
export TINKER_API_KEY="$(bash -ic 'printf %s "$TINKER_API_KEY"' 2>/dev/null)"
[ -n "$TINKER_API_KEY" ] || { echo "no TINKER_API_KEY"; exit 1; }
unset TINKER_BASE_URL TTD_M0_BASE_URL

# Grading registry: the 6 gradefarm hosts through the local SSH forwards (tunnel_loop.sh).
python3 - "$REG" <<'PY'
import json, sys, time
farms = [dict(name=f'gradefarm-harish-{i}', url=f'http://127.0.0.1:2786{i}', ok=True, tunnel_ok=True, health='healthy')
         for i in range(1, 7)]
import os; tmp = sys.argv[1] + '.tmp%d' % os.getpid(); json.dump(dict(time=time.time(), farms=farms), open(tmp, 'w')); os.replace(tmp, sys.argv[1])
PY

export PYTHONPATH="$DISC:$WT:$WT/tpu"
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/.cache/hf UV_CACHE_DIR=/scratch/gpfs/ZHUANGL/sk7524/.cache/uv
export WANDB_MODE=offline WANDB_PROJECT=tinker-inkling-erdos WANDB_CACHE_DIR=/scratch/gpfs/ZHUANGL/sk7524/.cache/wandb
export TTD_RUN_DIR="$RUN" EXPERIMENT_NAME="$NAME" SKYRL_RUN_ID="$NAME"

# --- model / sampling (Inkling) ---
export TTD_ENSEMBLE_MODELS="thinkingmachines/Inkling-2-Small:tml_v0:inkling" TTD_ALLOW_SINGLE_MEMBER=1
export TTD_INKLING_EFFORT="${TTD_INKLING_EFFORT:-0.75}" TTD_ANSWER_ONLY_CODE=1 TTD_ANSWER_MODEL_FAMILY=inkling
export TTD_NATIVE_THINKING_BUDGET=0 TTD_QWEN_TWO_PHASE=0 TTD_MIN_THINK_TOKENS=0
# fixed budget for every prompt: 24k thinking + 8k answer = 32k generated; 40k context fits prompts up to ~8.9k
# (a longer prompt shrinks thinking to fit: TTD_FIXED_BUDGET_FIT=1)
export CONTEXT_WINDOW=40960 TTD_M0_CONTEXT_WINDOW=40960 TTD_M0_TRAIN_MAX_SEQ=40960 TTD_M0_PHASE1_MAX_TOKENS=32768
export TTD_FIXED_THINK_TOKENS=24000 TTD_FIXED_TOTAL_TOKENS=32000 TTD_FIXED_BUDGET_FIT=1

# --- training (same as the Erdős runs except lr) ---
export GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-16}" GROUP_SIZE="${GROUP_SIZE:-32}" NUM_EPOCHS="$STEPS" SAVE_EVERY=1
export LEARNING_RATE=1e-5 LORA_RANK=32 KL_PENALTY_COEF=0 TTD_KL_MEASURE_EVERY=0 TTD_WARMUP_FB=0
export TTD_LOSS_FN=importance_sampling TTD_ADV_ESTIMATOR=piecewise_valid_entropic_centered TTD_ADV_PIECEWISE_ALPHA=1.0
export TTD_REJECT_TRUNCATED=1 TTD_EXPLICIT_LORA_SEED=1 TTD_M0_LORA_SEED="$LORA_SEED" TTD_INIT_STATE_SEED="$LORA_SEED"
export TTD_ERDOS_RECORD=0.3808563 TTD_ERDOS_CVXPY_SOLVERS="CLARABEL, SCS, OSQP, HIGHS, SCIPY"
export TTD_ELITE_SLOTS=2 TTD_ERDOS_PROMPT=v2 TTD_CROSS_WEIGHT=0 TTD_FLATLINE_STOP=0 TTD_RESTART_RATIO=0
export TTD_FB_GROUPS_PER_REQUEST=4 TTD_OPTIM_EARLY=1
export TTD_LEAGUE_PIPELINE=1 TTD_RESUME_STRICT=1 TTD_MAX_CONSEC_TRAIN_ERR=1 TTD_DISABLE_WANDB_TABLES=1
export TTD_SAMPLING_PROGRESS_TIMEOUT=-1 TTD_ENV=erdos_min_overlap TTD_PROBLEM_TYPE= TTD_FCALGO_MAX_CASES=0

# --- grading: gradefarm only, seeded, 1000 s, prompt says 1 CPU ---
export TTD_SAFE_GRADE_MAX_WORKERS=512  # client-side grading concurrency (default 64 throttled probe2 to ~8 waves)
export TTD_EVAL_BACKEND=farm EVAL_TIMEOUT="${EVAL_TIMEOUT:-1100}" NUM_CPUS_PER_TASK=1 TTD_RAY_PAYLOAD=0
export TTD_GRADE_SEED="$(python3 -c "import hashlib,sys; print(int(hashlib.sha256(sys.argv[1].encode()).hexdigest()[:8],16) % 2**31)" "$NAME")"
export TTD_GRADE_BUDGET_S="${TTD_GRADE_BUDGET_S:-1000}" TTD_ERDOS_PROMPT_CPUS=1
# How long a rollout waits for its grade (queue + run) before it counts as a 0-reward failure. The 8000 s default
# can expire in a long queue (15 clients x 512 programs on ~1000 slots) and would train on fake zeros; 48 h means it
# never fires in practice. A lost request is handled by the watchdog instead (restart after 6 h with grades pending).
export TTD_GRADE_WAIT_TIMEOUT=172800
unset SKYRL_GRADING_URL
export SKYRL_GRADING_LOCAL_SLOTS=0 SKYRL_OPEN_GRADING=1 SKYRL_GRADING_REGISTRY="$REG" SKYRL_OPEN_GRADING_K=6
export SKYRL_GRADING_FAMILIES='{"math": {"cpus": 1, "memory_gib": 1, "memory_max_gib": 2, "slots_per_host": 1}}'
export SKYRL_GRADING_MAX_INFRA_RETRIES=8 SKYRL_GRADING_LONG_POLL_SECONDS=20
export SKYRL_GRADING_JOURNAL="$RUN/grading-journal.jsonl" SKYRL_GRADING_EVENTS="$RUN/grading-events.jsonl"

env | grep -E '^(TTD_|GROUP|NUM_|LEARNING|LORA|CONTEXT|EVAL|SKYRL_|KL_)' | sort > "$RUN/env.txt"
echo "CODE_SKYRL=$(readlink -f "$WT")" >> "$RUN/env.txt"; echo "CODE_DISCOVER=$(readlink -f "$DISC")" >> "$RUN/env.txt"
echo "run $NAME steps $STEPS lora_seed $LORA_SEED grade_seed $TTD_GRADE_SEED -> $RUN"
cd "$WT"
# Diagnostics: tracebacks on native crashes / fatal signals, unbuffered output, and the exit status recorded.
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export DIAG_DIR="$RUN/diag"; mkdir -p "$DIAG_DIR"
set +e
# Forensics wrapper (same client; logs signals, kills, exits, heartbeats to $DIAG_DIR).
"$PY" -X faulthandler -W ignore::DeprecationWarning /scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops/diag_run.py &
PID=$!
echo "[launch_run] client pid $PID (pgid $(ps -o pgid= -p $PID | tr -d ' '), sid $(ps -o sid= -p $PID | tr -d ' '))"
# strace: every signal delivered to any thread (with sender pid/uid for catchable ones) and how the process ends.
strace -f -qq -e trace=none -e signal=all -o "$DIAG_DIR/strace.log" -p "$PID" 2>"$DIAG_DIR/strace.err" &
SPID=$!
# Outside view every 10 s: state, rss, threads; records the last time the process was seen alive.
( while kill -0 "$PID" 2>/dev/null; do
    echo "$(date -u +%FT%TZ) $(ps -o stat=,etime=,rss=,nlwp= -p "$PID" 2>/dev/null)" >> "$DIAG_DIR/ps.log"; sleep 10
  done; echo "$(date -u +%FT%TZ) gone" >> "$DIAG_DIR/ps.log" ) &
wait "$PID"
rc=$?
echo "[launch_run] client exited rc=$rc at $(date -u +%FT%TZ) (128+N = killed by signal N)"
sleep 2; kill "$SPID" 2>/dev/null
exit $rc
