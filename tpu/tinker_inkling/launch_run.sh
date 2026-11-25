#!/usr/bin/env bash
# Launch one Inkling-2-Small Erdős run on Tinker (client on della-vis1, grading on the gradefarm hosts).
# Usage: launch_run.sh RUN_NAME NUM_STEPS [LORA_SEED]
# Settings mirror the live Erdős merge runs (profile rebut-merge-erdos-muse-a client env) except the Tinker/Inkling
# specifics: model, renderer + effort, client-side thinking cap, lr 2e-5 (v3, 2026-11-17), gradefarm-only seeded 1000 s grading.
set -euo pipefail
{  # whole script in one block: bash parses it all before running, so replacing this file mid-run (deploy) is safe
NAME=$1; STEPS=$2; LORA_SEED=${3:-1}
CODE=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/code
PY=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/venv/bin/python
RUN=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling-runs/$NAME
REG=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops/registry.json
mkdir -p "$RUN"
# Code pinning: the first launch records the deployed snapshots in $RUN/code.pin; every later (re)start of this run
# execs the PINNED copy of this script with the pinned client code, so a deploy never changes a running experiment.
# (A clone gets its source seed's pin from grid.py.) Delete code.pin only to move a run to new code on purpose.
PIN="$RUN/code.pin"
if [ -z "${LAUNCH_PINNED:-}" ]; then
  if [ ! -f "$PIN" ]; then
    REL=$(readlink -f "$CODE/current")   # one read of the release link: skyrl and discover always from the same deploy
    printf 'PIN_SKYRL=%s\nPIN_DISCOVER=%s\n' "$(readlink -f "$REL/skyrl")" "$(readlink -f "$REL/discover")" > "$PIN.tmp"
    mv "$PIN.tmp" "$PIN"
  fi
  # shellcheck disable=SC1090
  . "$PIN"
  [ -f "$PIN_SKYRL/tpu/tinker_inkling/launch_run.sh" ] && [ -d "$PIN_DISCOVER/ttt_discover" ] \
    || { echo "[launch_run] pinned code missing: $PIN_SKYRL / $PIN_DISCOVER"; exit 1; }
  echo "[launch_run] code pinned: $PIN_SKYRL $PIN_DISCOVER"
  export LAUNCH_PINNED=1 PIN_SKYRL PIN_DISCOVER
  exec /bin/bash "$PIN_SKYRL/tpu/tinker_inkling/launch_run.sh" "$@"
fi
WT=$PIN_SKYRL
DISC=$PIN_DISCOVER

# Tinker key from ~/.bashrc (never printed).
# TINKER_KEY=2 (user 2026-11-25): the second key (TINKER_API_KEY_2, the user's other org) -- its own free-tier training
# slots. A run's checkpoints live in the org that wrote them, so a run must keep the same key for its whole life.
if [[ "${TINKER_KEY:-1}" =~ ^[2-9]$ ]]; then  # TINKER_KEY=N -> TINKER_API_KEY_N (key 3 added 2026-11-25)
  export TINKER_API_KEY="$(bash -ic "printf %s \"\$TINKER_API_KEY_${TINKER_KEY}\"" 2>/dev/null)"
else
  export TINKER_API_KEY="$(bash -ic 'printf %s "$TINKER_API_KEY"' 2>/dev/null)"
fi
[ -n "$TINKER_API_KEY" ] || { echo "no TINKER_API_KEY"; exit 1; }
unset TINKER_BASE_URL TTD_M0_BASE_URL

# Grading registry. Default: the 6 gradefarm hosts through the local SSH forwards (tunnel_loop.sh).
# GRADE_FARMS=ink (user 2026-11-25): the dedicated Inkling v6e-128 bf0y farms (built by the train-faster session,
# hash 2bb4ee34, listed only here) through tunnel_ink.sh's forwards 127.0.0.1:278NN, one per farm in ink-farms.json.
GRADE_FARMS="${GRADE_FARMS:-gradefarm}"
if [[ "$GRADE_FARMS" == ink ]]; then
  REG=/scratch/gpfs/ZHUANGL/sk7524/tinker-inkling/ops/registry-ink.json
  INK_FARMS_JSON=/scratch/gpfs/ZHUANGL/sk7524/rq4-local/tvg/inkling/ink-farms.json
  [ -f "$INK_FARMS_JSON" ] || { echo "refused: $INK_FARMS_JSON missing"; exit 2; }
fi
python3 - "$REG" "$GRADE_FARMS" "${INK_FARMS_JSON:-}" <<'PY'
import json, os, sys, time
reg, which = sys.argv[1], sys.argv[2]
if which == 'ink':
    farms = json.load(open(sys.argv[3]))
    farms = farms.get('farms', farms) if isinstance(farms, dict) else farms
    farms = [dict(name=f['name'], url=f'http://127.0.0.1:{27870 + i}', ok=True, tunnel_ok=True, health='healthy')
             for i, f in enumerate(sorted(farms, key=lambda f: f['name']), start=1)]
else:
    farms = [dict(name=f'gradefarm-harish-{i}', url=f'http://127.0.0.1:2786{i}', ok=True, tunnel_ok=True, health='healthy')
             for i in range(1, 7)]
tmp = reg + '.tmp%d' % os.getpid(); json.dump(dict(time=time.time(), farms=farms), open(tmp, 'w')); os.replace(tmp, reg)
print(f'[launch_run] grading registry {reg}: {len(farms)} farms')
PY
NFARMS=$(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))['farms']))" "$REG")

export PYTHONPATH="$DISC:$WT:$WT/tpu"
export HF_HOME=/scratch/gpfs/ZHUANGL/sk7524/.cache/hf UV_CACHE_DIR=/scratch/gpfs/ZHUANGL/sk7524/.cache/uv
export WANDB_MODE=offline WANDB_PROJECT=tinker-inkling-erdos WANDB_CACHE_DIR=/scratch/gpfs/ZHUANGL/sk7524/.cache/wandb
export TTD_RUN_DIR="$RUN" EXPERIMENT_NAME="$NAME" SKYRL_RUN_ID="$NAME"

# --- model / sampling (Inkling) ---
export TTD_ENSEMBLE_MODELS="thinkingmachines/Inkling-2-Small:tml_v0:inkling" TTD_ALLOW_SINGLE_MEMBER=1
export TTD_INKLING_EFFORT="${TTD_INKLING_EFFORT:-0.8}" TTD_ANSWER_ONLY_CODE=1 TTD_ANSWER_MODEL_FAMILY=inkling
export TTD_NATIVE_THINKING_BUDGET=0 TTD_QWEN_TWO_PHASE=0 TTD_MIN_THINK_TOKENS=0
# v3 budget (user 2026-11-17, after v2 stalled with thinking squeezed to ~15k): a FIXED 24,000-token thinking budget
# plus a 10,000-token answer (34,000 generated) on top of the prompt, in a 65,536 (64k) context. Prompts up to ~31k fit with
# full thinking; only a longer prompt shrinks thinking (FIT=1), never the answer. (v2 was 32,768 total, 10k answer.)
export CONTEXT_WINDOW=65536 TTD_M0_CONTEXT_WINDOW=65536 TTD_M0_TRAIN_MAX_SEQ=65536 TTD_M0_PHASE1_MAX_TOKENS=65536
export TTD_FIXED_TOTAL_TOKENS=34000 TTD_FIXED_THINK_TOKENS=24000 TTD_FIXED_BUDGET_FIT=1
# answer rules in the prompt (code only, compact, solver note; client env.py): v2 only. v3 = pilot prompt (off).
export TTD_ERDOS_ANSWER_RULES=0

# --- TTD_MODEL=gptoss (user 2026-11-18): gpt-oss-120b, the 32K-context variant ONLY (train $0.737/M; the
# :peft:131072 variant trains at $2.33/M), TTT-Discover's own setup: "high" reasoning, prompt + thinking <= 26,000
# tokens (phase 1), the answer gets the rest of the 32,768 context (>= ~6.7k; phase 2 forces the final channel when
# thinking runs out). No fixed think/answer budget (that is the Inkling completer's). Code only from the final channel.
if [[ "${TTD_MODEL:-inkling}" == gptoss ]]; then
  export TTD_ENSEMBLE_MODELS="openai/gpt-oss-120b:gpt_oss_high_reasoning:gptoss" TTD_ANSWER_MODEL_FAMILY=gptoss
  export CONTEXT_WINDOW=32768 TTD_M0_CONTEXT_WINDOW=32768 TTD_M0_TRAIN_MAX_SEQ=32768 TTD_M0_PHASE1_MAX_TOKENS=26000
  export TTD_FIXED_TOTAL_TOKENS=0 TTD_FIXED_THINK_TOKENS=0 TTD_FIXED_BUDGET_FIT=0
  unset TTD_INKLING_EFFORT
  [[ "$TTD_ENSEMBLE_MODELS" == openai/gpt-oss-120b:* && "$CONTEXT_WINDOW" == 32768 ]] || { echo "refused: gpt-oss must be the 32K model"; exit 2; }
# --- TTD_MODEL=mini (user 2026-11-25): Inkling-2-Mini, the FREE Tinker variant ONLY ($0 sample/train, 64K context,
# rate-limited; the paid ids bill $0.50/M sample). Same tml_v0 format, completer and budgets as Inkling-2-Small
# (24k thinking + 10k answer, 65,536 context = the free variant's max). '|' separates the spec because the id has ':'.
elif [[ "${TTD_MODEL:-inkling}" == mini ]]; then
  export TTD_ENSEMBLE_MODELS="thinkingmachines/Inkling-2-Mini:peft:65536:free|tml_v0|inkling"
  [[ "$TTD_ENSEMBLE_MODELS" == "thinkingmachines/Inkling-2-Mini:peft:65536:free|"* ]] || { echo "refused: mini must be the free variant"; exit 2; }
elif [[ "${TTD_MODEL:-inkling}" != inkling ]]; then
  echo "refused: unknown TTD_MODEL=$TTD_MODEL"; exit 2
fi

# --- training (same as the Erdős runs except lr) ---
export GROUPS_PER_BATCH="${GROUPS_PER_BATCH:-16}" GROUP_SIZE="${GROUP_SIZE:-32}" NUM_EPOCHS="$STEPS" SAVE_EVERY=1
export LEARNING_RATE="${LEARNING_RATE:-2e-5}" LORA_RANK=32 KL_PENALTY_COEF="${KL_PENALTY_COEF:-0}" TTD_KL_MEASURE_EVERY=0 TTD_WARMUP_FB=0
export TTD_LOSS_FN=importance_sampling TTD_ADV_ESTIMATOR=piecewise_valid_entropic_centered TTD_ADV_PIECEWISE_ALPHA=1.0
# v3 = pilot behaviour: zero-reward rollouts cut off at the budget are dropped from training (v2 trained them at 0)
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
export SKYRL_GRADING_LOCAL_SLOTS=0 SKYRL_OPEN_GRADING=1 SKYRL_GRADING_REGISTRY="$REG" SKYRL_OPEN_GRADING_K="$NFARMS"
export SKYRL_GRADING_FAMILIES='{"math": {"cpus": 1, "memory_gib": 1, "memory_max_gib": 2, "slots_per_host": 1}}'

# --- TTD_RECIPE=tvg (user 2026-11-25): the train-faster tvg Qwen/Muse Erdős recipe (profiles rebut-seedctl-erdos-qwen-s1,
# rebut-v6e128-erdos-muse-seed1, gen-profiles tvg-erdos-*-gen-s*): 28,672 context, prompt + thinking <= 22,528 (phase 1),
# answer = the rest; prompt v2 with ITS defaults (record 0.3808586, 2 CPUs, default solver line, no seed rule); plain
# run() grading (no seed/budget wrapper), 2 CPUs per program, no evaluate_erdos_solution alias; no elite slots. Model
# specifics (Inkling renderer + effort) and lr/KL/LoRA come from the run's own settings.
# TTD_RECIPE=tvg-a10k (user 2026-11-25): tvg with a 10k answer -- thinking budget unchanged (prompt + thinking <= 22,528),
# context 32,768 so the answer gets ~10.2k instead of ~6.1k (Inkling programs outgrew 6k: 45% cut off by step 15) -- and
# cut-off rollouts TRAIN as reward-0 failures (TTD_REJECT_TRUNCATED=0) so the model gets feedback for over-long answers.
if [[ "${TTD_RECIPE:-}" == tvg || "${TTD_RECIPE:-}" == tvg-a10k ]]; then
  export CONTEXT_WINDOW=28672 TTD_M0_CONTEXT_WINDOW=28672 TTD_M0_TRAIN_MAX_SEQ=28672 TTD_M0_PHASE1_MAX_TOKENS=22528
  if [[ "$TTD_RECIPE" == tvg-a10k ]]; then
    export CONTEXT_WINDOW=32768 TTD_M0_CONTEXT_WINDOW=32768 TTD_M0_TRAIN_MAX_SEQ=32768 TTD_REJECT_TRUNCATED=0
  fi
  export TTD_FIXED_TOTAL_TOKENS=0 TTD_FIXED_THINK_TOKENS=0 TTD_FIXED_BUDGET_FIT=0
  unset TTD_ERDOS_RECORD TTD_ERDOS_CVXPY_SOLVERS TTD_GRADE_SEED TTD_GRADE_BUDGET_S
  export TTD_ERDOS_PROMPT_CPUS=2 NUM_CPUS_PER_TASK=2 TTD_ELITE_SLOTS=0 TTD_ERDOS_EVAL_ALIAS=0 TTD_STATE_PROMPT_LEGACY=1
  export SKYRL_GRADING_FAMILIES='{"math": {"cpus": 2, "memory_gib": 4, "memory_max_gib": 8, "slots_per_host": 1}}'
elif [[ -n "${TTD_RECIPE:-}" ]]; then
  echo "refused: unknown TTD_RECIPE=$TTD_RECIPE"; exit 2
fi
export SKYRL_GRADING_MAX_INFRA_RETRIES=8 SKYRL_GRADING_LONG_POLL_SECONDS=20
export SKYRL_GRADING_JOURNAL="$RUN/grading-journal.jsonl" SKYRL_GRADING_EVENTS="$RUN/grading-events.jsonl"

env | grep -E '^(TTD_|GROUP|NUM_|LEARNING|LORA|CONTEXT|EVAL|SKYRL_|KL_)' | sort > "$RUN/env.txt"
echo "CODE_SKYRL=$(readlink -f "$WT")" >> "$RUN/env.txt"; echo "CODE_DISCOVER=$(readlink -f "$DISC")" >> "$RUN/env.txt"
echo "run $NAME steps $STEPS lora_seed $LORA_SEED grade_seed ${TTD_GRADE_SEED:-none (plain run())} -> $RUN"
cd "$WT"
# Diagnostics: tracebacks on native crashes / fatal signals, unbuffered output, and the exit status recorded.
export PYTHONFAULTHANDLER=1 PYTHONUNBUFFERED=1
export DIAG_DIR="$RUN/diag"; mkdir -p "$DIAG_DIR"
set +e
# Forensics wrapper (same client; logs signals, kills, exits, heartbeats to $DIAG_DIR).
"$PY" -X faulthandler -W ignore::DeprecationWarning "$WT/tpu/tinker_inkling/diag_run.py" &
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
}
