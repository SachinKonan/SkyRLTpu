#!/usr/bin/env bash
# Pass 2 of seed parity: grade the SPLASH candidates on the same judge chip.
#
# WHY A SECOND PASS. A judge actor boots for ONE problem (baseline election
# and noise-floor calibration are problem-specific), and only one judge chip
# landed in the drought, so pass 1 booted for rg_lru and every splash item
# was judge-faulted. Rather than wait for a second chip that may never come,
# re-point the same fleet at splash once rg_lru has drained. The reward cache
# keeps pass 1's verdicts, so nothing is regraded.
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$REPO"

PASS1_JOB="${PASS1_JOB:?set PASS1_JOB (the parity submitter slurm job id)}"
FLEET_JOB="${FLEET_JOB:?set FLEET_JOB (the judge fleet slurm job id)}"
QR_PREFIX="${QR_PREFIX:-sk7524-seedjudge-v6e}"

echo "=== waiting for pass 1 (rg_lru) to finish: job ${PASS1_JOB} ==="
while squeue -h -j "$PASS1_JOB" -o "%T" 2>/dev/null | grep -q .; do sleep 120; done
echo "=== pass 1 done $(date +%H:%M:%S) ==="
cp -f runs/pallas_arena/seed-parity-results.json \
      runs/pallas_arena/seed-parity-results-rglru.json 2>/dev/null || true

echo "=== re-pointing the fleet at splash_attention ==="
scancel "$FLEET_JOB" 2>/dev/null
sleep 20
new_fleet=$(sbatch --export=ALL,ZONE=us-east5-b,OTHER_ZONE=us-east5-a,QR_PREFIX="${QR_PREFIX}",N=2,ACCEL=v6e-1,RUNTIME=v2-alpha-tpuv6e,RAY_CHIPS=1,RAY_ACTORS=1,PROBLEMS=splash_attention,CACHE=gs://sk7524-pallas-arena-us-east5/reward-cache-seedparity-v6e-v1,IDLE_EXIT_S=3600,MIN_JUDGES=1 \
  -t 8:00:00 --job-name=pallas-judges-splash tpu/pallas_arena/rl_judges.sbatch | awk '{print $4}')
echo "=== splash fleet: ${new_fleet}; waiting for its queue ==="
rm -f runs/pallas_arena/rl-queue-url.txt
for _ in $(seq 1 240); do
  [ -s runs/pallas_arena/rl-queue-url.txt ] && break
  sleep 60
done
[ -s runs/pallas_arena/rl-queue-url.txt ] || { echo "FATAL: splash fleet never published a queue"; exit 1; }
url=$(cat runs/pallas_arena/rl-queue-url.txt)
echo "=== splash queue: ${url}; submitting splash candidates ==="

# The submitter must run where the queue is reachable (compute nodes only --
# the login node is firewalled off from compute-node ports).
sbatch -p all -N1 -n1 --cpus-per-task=4 --mem=8G -t 8:00:00 --job-name=seed-parity-splash \
  -o runs/pallas_arena/seed-parity-splash-%j.log \
  --wrap="cd $REPO && ARENA_PARITY_ONLY=splash_attention ARENA_PARITY_OUT=$REPO/runs/pallas_arena/seed-parity-results-splash.json \
    python3 tpu/pallas_arena/verify/submit_seed_parity.py" | tail -1
