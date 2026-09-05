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

# ADOPT AN IN-FLIGHT JOB. A supervisor normally starts by submitting, so
# restarting one while its cell already has a job running would duplicate the
# work and burn a second slice during a capacity drought. INITIAL_JOB lets a
# replacement supervisor pick up the existing job instead -- needed whenever
# the supervisor script itself is edited, since bash reads scripts lazily and
# running copies must be replaced.
job="${INITIAL_JOB:-}"
[ -n "$job" ] && echo "[supervisor:${TAG}] adopting in-flight job ${job}"
while true; do
  if [ -n "$job" ]; then
    gens="runs/pallas_arena/evolve-smoke-gens-${job}.jsonl"
    if squeue -h -j "$job" -o %T 2>/dev/null | grep -q .; then
      sleep 300; continue
    fi
    # SUCCESS MEANS USABLE GENERATIONS, NOT A NON-EMPTY FILE. This used to
    # test [ -s "$gens" ], so a cell whose every row was
    # {"error": "Connection refused"} counted as "gens landed" and the
    # supervisor exited satisfied. That is how a gemma cell with 10 of 32
    # usable was recorded as complete, and how a qwen cell would have been
    # accepted after its slice was preempted mid-generation (2026-08-27).
    # A file full of errors is worse than no file, because it looks like data.
    if [ -s "$gens" ]; then
      read -r n_ok n_tot <<< "$(python3 - "$gens" <<'PYEOF'
import json, sys
ok = tot = 0
for line in open(sys.argv[1]):
    line = line.strip()
    if not line:
        continue
    tot += 1
    try:
        r = json.loads(line)
    except Exception:
        continue
    if not r.get("error") and (r.get("text") or "").strip():
        ok += 1
print(ok, tot)
PYEOF
)"
      want=$(( ${GROUP_SIZE:-32} * ${MIN_USABLE_PCT:-75} / 100 ))
      if [ "${n_ok:-0}" -ge "$want" ]; then
        echo "[supervisor:${TAG}] gens landed: $gens ${n_ok}/${n_tot} usable $(date +%H:%M:%S)"
        echo "$job" > "runs/pallas_arena/bench-${TAG}-final-job.txt"
        exit 0
      fi
      echo "[supervisor:${TAG}] job $job produced only ${n_ok}/${n_tot} usable (want >= ${want}) $(date +%H:%M:%S); resubmitting"
      mv "$gens" "${gens%.jsonl}-rejected-${n_ok}of${n_tot}.jsonl" 2>/dev/null
    else
      echo "[supervisor:${TAG}] job $job died without gens $(date +%H:%M:%S); resubmitting"
    fi
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
