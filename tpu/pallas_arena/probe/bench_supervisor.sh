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

# ZONE ROTATION. Spot droughts are per-zone, not global: on 2026-08-27
# us-east5-a held six queued jobs (ours and other users') stuck in
# WAITING_FOR_RESOURCES for hours while us-east5-b was landing slices, and
# both rg_lru arms burned a full LAND_DEADLINE in zone a, twice, without
# ever asking b. Each attempt now flips the zone (and the complement zone
# used for teardown verification), so a drought in one costs one attempt
# instead of every attempt.
ZONES=(${BENCH_ZONES:-us-east5-a us-east5-b})
attempt=0

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
  export ZONE="${ZONES[$(( attempt % ${#ZONES[@]} ))]}"
  export OTHER_ZONE="${ZONES[$(( (attempt + 1) % ${#ZONES[@]} ))]}"
  attempt=$(( attempt + 1 ))
  job=$(sbatch --export=ALL --job-name="bench-${TAG}" \
          tpu/pallas_arena/probe/evolve_smoke.sbatch 2>/dev/null | awk '{print $4}')
  [ -n "$job" ] || { echo "[supervisor:${TAG}] sbatch failed; retry in 600s"; sleep 600; continue; }
  echo "[supervisor:${TAG}] submitted $job in ${ZONE} $(date +%H:%M:%S)"
  sleep 300
done
