#!/usr/bin/env bash
# Grade the prompt-v2 2x2 against a judge queue.
#
# Three things this does differently from grade_clean_2x2.sh:
#
#  * --seed-file is passed, and is REQUIRED here. The v2 prompt invites
#    `from lib import ...`; the judge knows nothing about lib, so the grading
#    client has to splice the seed's definitions back in first. Without it
#    every candidate that accepts the invitation dies at pregate on a missing
#    module -- the feature would look like a model failure.
#
#  * Job ids are READ, not hardcoded. The baseline driver pinned four job
#    numbers, which is wrong the moment a supervisor resubmits after a
#    preemption (it did, five times, on 2026-08-27/28). The supervisor writes
#    the winning job to bench-<tag>-final-job.txt only after verifying the
#    file has enough usable rows, so that file -- not a number typed in
#    advance -- is what identifies a cell's generations.
#
#  * Cells are graded AS THEY LAND, still strictly one at a time. Generation
#    (serving slices) and grading (judge slice) use different hardware, so
#    there is no reason to wait for all four; but two graders on one judge
#    host would put two candidates on the same chips and corrupt the wallclock
#    ratio that is the measurement (open task #18).
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"
S=$REPO/tpu/pallas_arena/probe

URL_FILE="${URL_FILE:-runs/pallas_arena/rl-queue-url.txt}"
MAX_WAIT_S="${MAX_WAIT_S:-14400}"
DEADLINE_S="${DEADLINE_S:-172800}"          # 48h: arms queue for spot for hours

# tag : task : seed file
CELLS=(
  "qwen-p2-splash:splash_attention:$S/seed_splash_flash.py"
  "gemma-p2-splash:splash_attention:$S/seed_splash_flash.py"
  "qwen-p2-rglru:rg_lru:$S/seed_rglru_active.py"
  "gemma-p2-rglru:rg_lru:$S/seed_rglru_active.py"
)

queue_url() {
  # A published URL is not a staffed queue: rl-queue-url.txt outlives the
  # fleet that wrote it and /status answers happily with zero judges attached,
  # which would return 32 rows of "no verdict before deadline" -- a
  # full-looking file with no measurement in it.
  local cand; cand=$(cat "$URL_FILE" 2>/dev/null)
  [ -n "$cand" ] || return 1
  python3 - "$cand" <<'PY' || return 1
import json, sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/status", timeout=10) as r:
        d = json.load(r)
except Exception:
    sys.exit(1)
seen = d.get("workers_seen") or {}
fresh = [w for w, a in seen.items() if isinstance(a, (int, float)) and a < 300]
sys.exit(0 if d.get("ok") and fresh else 1)
PY
  echo "$cand"
}

deadline=$(( $(date +%s) + DEADLINE_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  todo=0
  for spec in "${CELLS[@]}"; do
    IFS=: read -r tag task seed <<< "$spec"
    out="runs/pallas_arena/graded-p2-${tag}.json"
    [ -s "$out" ] && continue
    todo=$(( todo + 1 ))
    jobfile="runs/pallas_arena/bench-${tag}-final-job.txt"
    [ -s "$jobfile" ] || continue                      # cell still generating
    job=$(cat "$jobfile")
    gens="runs/pallas_arena/evolve-smoke-gens-${job}.jsonl"
    [ -s "$gens" ] || { echo "$(date +%H:%M:%S) [$tag] job $job has no gens file"; continue; }
    url=$(queue_url) || { echo "$(date +%H:%M:%S) waiting for a staffed judge queue"; break; }
    echo "$(date +%H:%M:%S) [$tag] grading $(wc -l < "$gens") candidates on $url"
    uv run --isolated --extra jax --with fastapi python \
      "$S/grade_gens_via_queue.py" --gens "$gens" --queue "$url" --out "$out" \
      --seed-file "$seed" --max-wait-s "$MAX_WAIT_S" \
      > "runs/pallas_arena/graded-p2-${tag}.log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "$(date +%H:%M:%S) [$tag] FAILED rc=$rc -- see graded-p2-${tag}.log"
      rm -f "$out"                                     # never leave a partial as done
    else
      echo "$(date +%H:%M:%S) [$tag] done"
      tail -2 "runs/pallas_arena/graded-p2-${tag}.log"
      # How much the lib feature was actually used -- the headline of this A/B.
      python3 - "$gens" "$seed" <<'PY'
import json, sys, pathlib
sys.path.insert(0, "tpu")
from pallas_arena.probe.gen_smoke import extract_completion
from pallas_arena.probe.lib_splice import wanted
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
used = tot = 0
for r in rows:
    if not r.get("text"):
        continue
    tot += 1
    p = extract_completion(r["text"], ["kernel"], family=r.get("family"))
    if p and wanted(p):
        used += 1
print(f"    lib imports used by {used}/{tot} candidates")
PY
    fi
    break                                              # one grading at a time
  done
  [ "$todo" -eq 0 ] && { echo "$(date +%H:%M:%S) ALL FOUR P2 CELLS GRADED"; exit 0; }
  sleep 120
done
echo "$(date +%H:%M:%S) deadline reached with cells ungraded"
exit 1
