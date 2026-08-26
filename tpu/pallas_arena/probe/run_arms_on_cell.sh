#!/usr/bin/env bash
# ARMS CLIENT -- runs ON w0 of the arm cell, in tmux (league pattern).
#
# Both problems in parallel: the rg_lru seeded arm generates against engine
# :8001 while the splash arm generates against :8002 (32 completions each,
# 32k ctx, think 10240 / answer 15360). Programs are then graded on REAL
# silicon through w1's queue (fwd+bwd, tp4 included) and everything --
# gens, graded verdicts -- is pushed to GCS for the login side to collect.
#
# Prereqs (arm_v6e16_bringup.sh did all of this): ~/arena tree, engines
# warming on :8001/:8002, ~/arena-queue-url.txt pointing at w1.
#
# Usage (tmux on w0):
#   ARM_TAG=qwen bash ~/arena/pallas_arena/probe/run_arms_on_cell.sh
set -uo pipefail

ARM_TAG="${ARM_TAG:?set ARM_TAG (qwen|gemma -- labels the outputs)}"
MODEL="${SERVED_MODEL:-$( [ "$ARM_TAG" = gemma ] && echo google/gemma-4-31B-it || echo Qwen/Qwen3.5-27B )}"
QUEUE_URL="$(cat ~/arena-queue-url.txt)"
OUT=~/arm-results
BUCKET=gs://sk7524-pallas-arena-us-east5/arm-results
mkdir -p "$OUT"
cd ~/arena

export PATH="$HOME/.local/bin:$PATH"
PY="$HOME/.venvs/vllm-tpu/bin/python"   # has urllib only needs stdlib; venv exists from serving
[ -x "$PY" ] || PY=python3

echo "=== waiting for both engines + queue $(date +%H:%M:%S) ==="
for dep in "http://127.0.0.1:8001/v1/models" "http://127.0.0.1:8002/v1/models" "${QUEUE_URL}/status"; do
  for i in $(seq 1 240); do
    curl -fsS -m5 "$dep" >/dev/null 2>&1 && { echo "up: $dep"; break; }
    [ "$i" = 240 ] && { echo "FATAL: never up: $dep"; exit 1; }
    sleep 20
  done
done

# Seeds + observations: seed_rglru_active.py is the silicon-verified winner
# (written by the seed-parity step); observations are the judge's real
# per-shape feedback. Fall back loudly if a piece is missing.
S=~/arena/pallas_arena/probe
RG_SEED="$S/seed_rglru_active.py"; [ -s "$RG_SEED" ] || RG_SEED="$S/seed_rglru_timeblocked.py"
SP_SEED="$S/seed_splash_flash.py"
RG_OBS="$S/../seed-obs-rglru.txt";  [ -s "$RG_OBS" ] || RG_OBS=""
SP_OBS="$S/../seed-obs-splash.txt"; [ -s "$SP_OBS" ] || SP_OBS=""
RG_REWARD="$(cat "$S/../seed-reward-rglru.txt" 2>/dev/null || echo '1.0x (parity assumed -- parity run pending)')"
SP_REWARD="$(cat "$S/../seed-reward-splash.txt" 2>/dev/null || echo '1.0x (parity assumed -- parity run pending)')"

gen() { # gen <cells> <port> <seed> <obs> <reward> <tag>
  local cells="$1" port="$2" seed="$3" obs="$4" reward="$5" tag="$6"
  $PY pallas_arena/probe/gen_smoke.py \
    --server "http://127.0.0.1:${port}" --model "$MODEL" \
    --out "$OUT/gens-${ARM_TAG}-${tag}.jsonl" --group-size "${GROUP_SIZE:-32}" \
    --max-tokens "${MAX_NEW_TOKENS:-10240}" --answer-cap "${ANSWER_CAP:-15360}" \
    --ctx 32768 --concurrency 6 --cells "$cells" \
    --seed-file "$seed" ${obs:+--seed-observation "$obs"} --seed-reward "$reward" \
    2>&1 | tee -a "$OUT/gen-${ARM_TAG}-${tag}.log"
}

echo "=== generating BOTH problems in parallel $(date +%H:%M:%S) ==="
gen "rg_lru:rf3s"           8001 "$RG_SEED" "$RG_OBS" "$RG_REWARD" rglru  &
P_RG=$!
gen "splash_attention:rf3s" 8002 "$SP_SEED" "$SP_OBS" "$SP_REWARD" splash &
P_SP=$!
wait "$P_RG"; rc_rg=$?
wait "$P_SP"; rc_sp=$?
echo "=== generation done: rglru rc=$rc_rg splash rc=$rc_sp $(date +%H:%M:%S) ==="

echo "=== grading on silicon via ${QUEUE_URL} ==="
for tag in rglru splash; do
  gens="$OUT/gens-${ARM_TAG}-${tag}.jsonl"
  [ -s "$gens" ] || { echo "no gens for $tag; skipping"; continue; }
  PYTHONPATH=~/arena $PY pallas_arena/probe/grade_gens_via_queue.py \
    --gens "$gens" --queue "$QUEUE_URL" \
    --out "$OUT/graded-${ARM_TAG}-${tag}.json" \
    2>&1 | tee -a "$OUT/grade-${ARM_TAG}-${tag}.log" &
done
wait

echo "=== pushing results to ${BUCKET}/${ARM_TAG}/ ==="
gsutil -m -q rsync -r "$OUT" "${BUCKET}/${ARM_TAG}/" && echo "results banked to GCS"
echo "=== arms complete $(date +%H:%M:%S) ==="
