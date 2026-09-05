#!/usr/bin/env bash
# Grade ONLY the cells that still need it, on the live judge queue. No skip
# logic (the recovery's sed-escaped check errored and flagged complete cells
# as incomplete, deleting them). Explicit list = exactly what to grade.
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; cd "$REPO"
export PYTHONPATH="$REPO/tpu:${PYTHONPATH:-}"
S=$REPO/tpu/pallas_arena/probe
URL_FILE="${URL_FILE:-runs/pallas_arena/rl-queue-url.txt}"
ITEMS=(
  # v1-qwen-splash: deleted by the broken recovery, regenerate it
  "runs/pallas_arena/evolve-smoke-gens-3761157.jsonl:runs/pallas_arena/xla-v1-qwen-splash.json:$S/seed_splash_flash.py"
  # the 4 v2 cells -- the actual goal
  "runs/pallas_arena/coserve-gens-qwen-rglru-3783640.jsonl:runs/pallas_arena/xla-v2-qwen-rglru.json:$S/seed_rglru_active.py"
  "runs/pallas_arena/coserve-gens-qwen-splash-3783639.jsonl:runs/pallas_arena/xla-v2-qwen-splash.json:$S/seed_splash_flash.py"
  "runs/pallas_arena/v6e-arm-gens-gemma-rglru.jsonl:runs/pallas_arena/xla-v2-gemma-rglru.json:$S/seed_rglru_active.py"
  "runs/pallas_arena/coserve-gens-gemma-splash-3783641.jsonl:runs/pallas_arena/xla-v2-gemma-splash.json:$S/seed_splash_flash.py"
)
queue_url() {
  local cand; cand=$(cat "$URL_FILE" 2>/dev/null); [ -n "$cand" ] || return 1
  python3 - "$cand" <<'PY' || return 1
import json,sys,urllib.request
try:
  d=json.load(urllib.request.urlopen(sys.argv[1].rstrip("/")+"/status",timeout=10))
except Exception: sys.exit(1)
fresh=[w for w,a in (d.get("workers_seen") or {}).items() if isinstance(a,(int,float)) and a<300]
sys.exit(0 if d.get("ok") and fresh else 1)
PY
  echo "$cand"
}
deadline=$(( $(date +%s) + ${READY_TIMEOUT_S:-86400} )); url=""
while [ "$(date +%s)" -lt "$deadline" ]; do url=$(queue_url) && break || true; sleep 60; done
[ -n "$url" ] || { echo "FATAL: no staffed judge"; exit 1; }
echo "$(date +%H:%M:%S) queue live: $url"
for spec in "${ITEMS[@]}"; do
  IFS=: read -r gens out seed <<< "$spec"; name=$(basename "$out" .json)
  # re-grade only if MISSING or has no-verdict rows; else skip (simple, robust)
  if [ -s "$out" ]; then
    if python3 -c "import json,sys;d=json.load(open(sys.argv[1]));r=d.get('rows') or [x for c in d.values() for x in c.get('rows',[])];sys.exit(0 if any('no verdict' in str(x.get('gate') or x.get('outcome') or '') for x in r) else 1)" "$out" 2>/dev/null; then
      echo "$(date +%H:%M:%S) [$name] partial -> re-grade"; rm -f "$out"
    else
      echo "$(date +%H:%M:%S) [$name] complete -> skip"; continue
    fi
  fi
  echo "$(date +%H:%M:%S) [$name] grading $(wc -l < "$gens") items"
  uv run --isolated --extra jax --with fastapi python "$S/grade_gens_via_queue.py" \
    --gens "$gens" --queue "$url" --out "$out" --seed-file "$seed" --max-wait-s 43200 \
    > "runs/pallas_arena/${name}.log" 2>&1 && echo "$(date +%H:%M:%S) [$name] done" || echo "$(date +%H:%M:%S) [$name] FAILED"
done
echo "$(date +%H:%M:%S) V2 REGRADE DONE"
