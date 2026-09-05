#!/usr/bin/env bash
# Grade the SEED programs on the same judge, case set, and protocol as the
# clean 2x2 cells.
#
# Why this exists: the bar quoted in the prompt (splash 0.262, rg_lru 1.000)
# was measured on an OLDER grading run whose scored set was 7 single-chip
# cases. The judge serving the 2x2 scores 10 -- those 7 plus tp4-gqa32x8-s4096,
# tp4-h32-s4096 and tp4-mqa-h32kv1-s4096. Reward is a geomean over the scored
# cases, so the two numbers are not comparable and "beat the seed" cannot be
# decided against the old one. Re-grading the seed here produces a bar on the
# IDENTICAL case set, which is the only bar the results may be read against.
#
# Runs strictly AFTER the cells (sbatch --dependency): one candidate on the
# same host, never beside them, so the seed's own timings are measured under
# the same non-contended conditions as the candidates' (open task #18).
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"

URL_FILE="${URL_FILE:-runs/pallas_arena/rl-queue-url.txt}"
MAX_WAIT_S="${MAX_WAIT_S:-14400}"
READY_TIMEOUT_S="${READY_TIMEOUT_S:-7200}"

CELLS=(
  "runs/pallas_arena/seedbar-gens-splash_attention.jsonl:runs/pallas_arena/graded-seedbar-splash.json"
  "runs/pallas_arena/seedbar-gens-rg_lru.jsonl:runs/pallas_arena/graded-seedbar-rglru.json"
)

deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
url=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  cand=$(cat "$URL_FILE" 2>/dev/null)
  if [ -n "$cand" ] && python3 - "$cand" <<'PY'
import json, sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/status", timeout=10) as r:
        d = json.load(r)
except Exception:
    sys.exit(1)
seen = d.get("workers_seen") or {}
fresh = [w for w, age in seen.items() if isinstance(age, (int, float)) and age < 300]
sys.exit(0 if d.get("ok") and fresh else 1)
PY
  then
    url="$cand"; break
  fi
  sleep 60
done
[ -n "$url" ] || { echo "$(date +%H:%M:%S) FATAL: judge gone before the seed bar could be measured"; exit 1; }
echo "$(date +%H:%M:%S) queue live: $url"

rc_all=0
for spec in "${CELLS[@]}"; do
  gens="${spec%%:*}"; out="${spec##*:}"
  name=$(basename "$out" .json)
  echo "$(date +%H:%M:%S) [$name] grading the seed"
  uv run --isolated --extra jax --with fastapi python \
    tpu/pallas_arena/probe/grade_gens_via_queue.py \
    --gens "$gens" --queue "$url" --out "$out" --max-wait-s "$MAX_WAIT_S" \
    > "runs/pallas_arena/${name}.log" 2>&1
  rc=$?
  [ $rc -eq 0 ] || { echo "$(date +%H:%M:%S) [$name] FAILED rc=$rc"; rc_all=1; }
  tail -2 "runs/pallas_arena/${name}.log"
done
echo "$(date +%H:%M:%S) SEED BAR DONE rc=$rc_all"
exit $rc_all
