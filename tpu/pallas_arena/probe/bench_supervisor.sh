#!/usr/bin/env bash
# Generic bench supervisor: resubmit one evolve_smoke bench through spot
# droughts until its generations land. All bench knobs (SMOKE_CELLS,
# QWEN_MODEL, REPAIR_FROM, ...) are inherited from the caller's exported
# environment; SUP_TAG names the lock/log/final-job artifacts.
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"
TAG="${SUP_TAG:?SUP_TAG required}"

LOCK="/tmp/bench-supervisor-${TAG}.lock"
exec 9>"$LOCK"
flock -n 9 || { echo "another supervisor holds $LOCK"; exit 0; }

job=""
while true; do
  if [ -n "$job" ]; then
    gens="runs/pallas_arena/evolve-smoke-gens-${job}.jsonl"
    if squeue -h -j "$job" -o %T 2>/dev/null | grep -q .; then
      sleep 300; continue
    fi
    if [ -s "$gens" ]; then
      echo "[supervisor:${TAG}] gens landed: $gens $(date +%H:%M:%S)"
      echo "$job" > "runs/pallas_arena/bench-${TAG}-final-job.txt"
      exit 0
    fi
    echo "[supervisor:${TAG}] job $job died without gens $(date +%H:%M:%S); resubmitting"
  fi
  job=$(sbatch --export=ALL --job-name="bench-${TAG}" \
          tpu/pallas_arena/probe/evolve_smoke.sbatch 2>/dev/null | awk '{print $4}')
  [ -n "$job" ] || { echo "[supervisor:${TAG}] sbatch failed; retry in 600s"; sleep 600; continue; }
  echo "[supervisor:${TAG}] submitted $job $(date +%H:%M:%S)"
  sleep 300
done
