#!/usr/bin/env bash
# Grade the clean 2x2 (qwen|gemma x splash|rg_lru) against a judge queue.
#
# Waits for the judge fleet to publish rl-queue-url.txt, then grades all FOUR
# cells through pallas_arena.probe.grade_gens_via_queue -- the SAME entrypoint
# and the SAME canonical extraction the RL loop uses, so nothing here is a
# report-only code path that can disagree with training.
#
# Cells are graded SEQUENTIALLY. They share one judge queue and one set of
# chips, and wallclock timing is the measurement: overlapping two cells would
# put two candidates on the same host and corrupt the very number we are
# collecting (open task #18).
#
# Idempotent: a cell whose output already exists and parses is skipped, so a
# rerun after a preemption resumes instead of re-grading.
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"

URL_FILE="${URL_FILE:-runs/pallas_arena/rl-queue-url.txt}"
MAX_WAIT_S="${MAX_WAIT_S:-14400}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-64800}"

# gens:out, in the order they were generated.
CELLS=(
  "runs/pallas_arena/evolve-smoke-gens-3761157.jsonl:runs/pallas_arena/graded-clean-qwen-splash.json"
  "runs/pallas_arena/evolve-smoke-gens-3761336.jsonl:runs/pallas_arena/graded-clean-qwen-rglru.json"
  "runs/pallas_arena/evolve-smoke-gens-3762410.jsonl:runs/pallas_arena/graded-clean-gemma-rglru.json"
  "runs/pallas_arena/evolve-smoke-gens-3763890.jsonl:runs/pallas_arena/graded-clean-gemma-splash.json"
)

for spec in "${CELLS[@]}"; do
  g="${spec%%:*}"
  [ -s "$g" ] || { echo "FATAL: missing generations $g"; exit 1; }
done

echo "$(date +%H:%M:%S) waiting for judge queue at $URL_FILE"
deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
url=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  cand=$(cat "$URL_FILE" 2>/dev/null)
  # A published URL is not a SERVING queue, and a reachable queue is not a
  # STAFFED one. rl-queue-url.txt outlives the fleet that wrote it, and the
  # queue answers /status happily with zero judges attached -- every candidate
  # would then sit unleased until --max-wait-s and come back as
  # "no verdict before deadline", i.e. a full-looking result file with no
  # measurements in it. Require a worker that has polled recently.
  if [ -n "$cand" ] && python3 - "$cand" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/status", timeout=10) as r:
        d = json.load(r)
except Exception:
    sys.exit(1)
seen = d.get("workers_seen") or {}
fresh = [w for w, age in seen.items() if isinstance(age, (int, float)) and age < 300]
print(f"  queue ok={d.get('ok')} depth={d.get('queue_depth')} "
      f"workers={len(seen)} fresh={len(fresh)}", file=sys.stderr)
sys.exit(0 if d.get("ok") and fresh else 1)
PY
  then
    url="$cand"; break
  fi
  sleep 60
done
[ -n "$url" ] || { echo "$(date +%H:%M:%S) FATAL: no healthy queue before deadline"; exit 1; }
echo "$(date +%H:%M:%S) queue live: $url"

rc_all=0
for spec in "${CELLS[@]}"; do
  gens="${spec%%:*}"; out="${spec##*:}"
  name=$(basename "$out" .json)
  if python3 -c "import json,sys; d=json.load(open(sys.argv[1])); sys.exit(0 if any(r.get('outcome') for c in d.values() for r in c.get('rows',[])) else 1)" "$out" 2>/dev/null; then
    echo "$(date +%H:%M:%S) [$name] already graded; skipping"
    continue
  fi
  echo "$(date +%H:%M:%S) [$name] grading $(wc -l < "$gens") candidates"
  uv run --isolated --extra jax --with fastapi python \
    tpu/pallas_arena/probe/grade_gens_via_queue.py \
    --gens "$gens" --queue "$url" --out "$out" --max-wait-s "$MAX_WAIT_S" \
    > "runs/pallas_arena/${name}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$(date +%H:%M:%S) [$name] FAILED rc=$rc -- see runs/pallas_arena/${name}.log"
    rc_all=1
  else
    echo "$(date +%H:%M:%S) [$name] done -> $out"
    tail -2 "runs/pallas_arena/${name}.log"
  fi
done
echo "$(date +%H:%M:%S) ALL CELLS ATTEMPTED rc=$rc_all"
exit $rc_all
