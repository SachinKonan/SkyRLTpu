#!/usr/bin/env bash
# Grade the banked V2 cells on a V6E judge -- the arena's first-class silicon.
#
# Decision 2026-09-01: grade on v6e, full stop. The arena was designed for a
# v6e-8 judge (TP-on-v6e-8 first-class), the v1 baseline and the seed bars
# (0.2316 / 1.0000) were ALREADY graded there by judge3, and v5p grading
# distorts the task itself -- kernels written against v6e's VMEM budget die at
# compile on v5p (15/32 VMEM stack OOMs on the first v5p-graded cell). So the
# only work is the banked v2 cells; the seeds run first as a cheap cross-judge
# reproducibility check against judge3's bars.
#
# One gens file at a time: a single judge host, and wallclock IS the metric.
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"
export PYTHONPATH="$REPO/tpu:${PYTHONPATH:-}"
URL_FILE="${URL_FILE:-runs/pallas_arena/rl-queue-url.txt}"
S=$REPO/tpu/pallas_arena/probe

# gens : out : seed-file(for lib splice; baseline gens have none, harmless)
ITEMS=(
  "runs/pallas_arena/seedbar-gens-rg_lru.jsonl:runs/pallas_arena/xla-seedcheck-rglru.json:$S/seed_rglru_active.py"
  "runs/pallas_arena/seedbar-gens-splash_attention.jsonl:runs/pallas_arena/xla-seedcheck-splash.json:$S/seed_splash_flash.py"
  # v1 BASELINE cells -- on disk they were graded vs production; regrade vs XLA
  # so v1 and v2 sit on the same denominator and the prompt A/B is readable.
  "runs/pallas_arena/evolve-smoke-gens-3761157.jsonl:runs/pallas_arena/xla-v1-qwen-splash.json:$S/seed_splash_flash.py"
  "runs/pallas_arena/evolve-smoke-gens-3761336.jsonl:runs/pallas_arena/xla-v1-qwen-rglru.json:$S/seed_rglru_active.py"
  "runs/pallas_arena/evolve-smoke-gens-3762410.jsonl:runs/pallas_arena/xla-v1-gemma-rglru.json:$S/seed_rglru_active.py"
  "runs/pallas_arena/evolve-smoke-gens-3763890.jsonl:runs/pallas_arena/xla-v1-gemma-splash.json:$S/seed_splash_flash.py"
  # The three banked coserve v2 cells, REgraded here so every dataset in the
  # A/B -- seeds, all v1 cells, all v2 cells -- carries verdicts from ONE
  # judge on one host. The w1 coserve verdicts (one complete cell, two
  # partials cut off by walltime) stay on disk as corroboration only.
  "runs/pallas_arena/coserve-gens-qwen-rglru-3783640.jsonl:runs/pallas_arena/xla-v2-qwen-rglru.json:$S/seed_rglru_active.py"
  "runs/pallas_arena/coserve-gens-qwen-splash-3783639.jsonl:runs/pallas_arena/xla-v2-qwen-splash.json:$S/seed_splash_flash.py"
  "runs/pallas_arena/coserve-gens-gemma-splash-3783641.jsonl:runs/pallas_arena/xla-v2-gemma-splash.json:$S/seed_splash_flash.py"
  "runs/pallas_arena/v6e-arm-gens-gemma-rglru.jsonl:runs/pallas_arena/xla-v2-gemma-rglru.json:$S/seed_rglru_active.py"
)

queue_url() {
  # A published URL is not a staffed queue: rl-queue-url.txt outlives its
  # fleet (it currently holds dead judge9), and /status answers with zero
  # judges attached. Require ok=true AND a worker that polled within 300s.
  local cand; cand=$(cat "$URL_FILE" 2>/dev/null); [ -n "$cand" ] || return 1
  python3 - "$cand" <<'PY' || return 1
import json, sys, urllib.request
try:
    with urllib.request.urlopen(sys.argv[1].rstrip("/") + "/status", timeout=10) as r:
        d = json.load(r)
except Exception:
    sys.exit(1)
fresh = [w for w, a in (d.get("workers_seen") or {}).items()
         if isinstance(a, (int, float)) and a < 300]
sys.exit(0 if d.get("ok") and fresh else 1)
PY
  echo "$cand"
}

deadline=$(( $(date +%s) + ${READY_TIMEOUT_S:-64800} ))
url=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  url=$(queue_url) && break || true
  sleep 60
done
[ -n "$url" ] || { echo "FATAL: no staffed judge before deadline"; exit 1; }
echo "$(date +%H:%M:%S) queue live: $url"

rc_all=0
for spec in "${ITEMS[@]}"; do
  IFS=: read -r gens out seed <<< "$spec"
  name=$(basename "$out" .json)
  [ -s "$out" ] && { echo "$(date +%H:%M:%S) [$name] already graded; skip"; continue; }
  [ -s "$gens" ] || { echo "$(date +%H:%M:%S) [$name] gens not yet generated; skip"; continue; }
  echo "$(date +%H:%M:%S) [$name] grading $(wc -l < "$gens") item(s)"
  uv run --isolated --extra jax --with fastapi python "$S/grade_gens_via_queue.py" \
    --gens "$gens" --queue "$url" --out "$out" --seed-file "$seed" --max-wait-s 14400 \
    > "runs/pallas_arena/${name}.log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "$(date +%H:%M:%S) [$name] FAILED rc=$rc"; rm -f "$out"; rc_all=1
  else
    echo "$(date +%H:%M:%S) [$name] done"; tail -2 "runs/pallas_arena/${name}.log"
  fi
done
echo "$(date +%H:%M:%S) REGRADES DONE rc=$rc_all"
exit $rc_all
