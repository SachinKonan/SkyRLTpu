#!/bin/bash
# Phase-5 GRADING-SERVER FLEET DEMONSTRATION (no RL, no training, no policy).
#
# Queue on THIS compute node; N spot v6e-1 judges in us-east5-b all polling
# that ONE queue through their own reverse ssh tunnels; a fixed corpus in,
# verdicts out. Judges are stateless pullers, so a preemption (or our own
# deliberate mid-lease SIGKILL) is a requeue, not a lost result.
#
# HARD RESOURCE RULES, in the code rather than in a comment:
#   * never more than N=5 arena QRs alive, named sk7524-pallas-judge-1..5 so
#     they are unmistakable next to anything else in the zone;
#   * every QR deleted on EVERY exit path -- success, failure, exception,
#     TERM/INT/HUP (slurm time limit, scancel, node drain), and the 3h cap.
#     Phase 4 found `trap cleanup EXIT` does NOT fire on an untrapped
#     SIGTERM, which is exactly what a slurm timeout sends; that hardening
#     is kept and now covers all five slots;
#   * a hard wall-clock cap measured from the FIRST QR create;
#   * if fewer than N land inside the landing deadline, run with what landed
#     (>=2), say so, and delete the ones still pending.
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH --job-name=pallas-arena-phase5
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/phase5-%j.log

set -uo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run under SLURM (sbatch); no compute on the login node." >&2
  exit 1
fi

N=${N:-5}
QR_PREFIX=${QR_PREFIX:-sk7524-pallas-judge}
ZONE=${ZONE:-us-east5-b}
PROJECT=${PROJECT:-vision-mix}
ACCEL=${ACCEL:-v6e-1}
RUNTIME=${RUNTIME:-v2-alpha-tpuv6e}
PORT=${PORT:-8791}
LAND_DEADLINE_S=${LAND_DEADLINE_S:-2700}     # 45 min to land 5 spot QRs
HARD_CAP_S=${HARD_CAP_S:-10800}              # 3 h from the first QR create
LEASE_TIMEOUT_S=${LEASE_TIMEOUT_S:-240}
TARGET=${TARGET:-1600}
DUPS=${DUPS:-100}
COMPILE_BUDGET_S=${COMPILE_BUDGET_S:-90}
CACHE=${CACHE:-gs://sk7524-pallas-arena-us-east5/reward-cache-phase5}
MIN_JUDGES=${MIN_JUDGES:-2}

repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
out_dir="${repo_root}/runs/pallas_arena"
work_dir="${out_dir}/phase5-work-${SLURM_JOB_ID}"
ready_dir="${work_dir}/ready"
mkdir -p "$out_dir" "$ready_dir"
cd "$repo_root"
export PYTHONPATH="${repo_root}/tpu:${PYTHONPATH:-}"

judges=()
for i in $(seq 1 "$N"); do judges+=("${QR_PREFIX}-${i}"); done

gc() { timeout 600 gcloud compute tpus "$@" --zone="$ZONE" --project="$PROJECT"; }
tssh() { timeout 900 gcloud compute tpus tpu-vm ssh "$1" --zone="$ZONE" --project="$PROJECT" --worker=0 --command="$2"; }

queue_pid=""
keeper_pid=""
watchdog_pid=""
sup_pids=()

# ------------------------------------------------------------------ cleanup
cleaned=""
cleanup() {
  [ -n "$cleaned" ] && return 0
  cleaned=1
  echo "[cleanup] $(date +%H:%M:%S) tearing the fleet down (always-delete)"
  for p in "${sup_pids[@]:-}" "$keeper_pid" "$watchdog_pid" "$queue_pid"; do
    [ -n "${p:-}" ] && kill "$p" 2>/dev/null
  done
  # kill any ssh/gcloud children the supervisors left holding tunnels
  pkill -P $$ 2>/dev/null
  sleep 1
  for name in "${judges[@]}"; do
    ( timeout 600 gcloud compute tpus queued-resources delete "$name" \
        --zone="$ZONE" --project="$PROJECT" --force --quiet >/dev/null 2>&1
      echo "[cleanup] delete issued: $name" ) &
  done
  wait
  # Verify, then delete anything of ours that survived. "Issued" is not
  # "gone"; the only acceptable end state is an empty list in BOTH zones.
  for z in us-east5-b us-east5-a; do
    left=$(timeout 300 gcloud compute tpus queued-resources list --zone="$z" --project="$PROJECT" \
             --format='value(name)' 2>/dev/null | grep -E "^${QR_PREFIX}-" || true)
    if [ -n "$left" ]; then
      echo "[cleanup] SURVIVORS in $z: $left -- deleting again"
      for name in $left; do
        timeout 600 gcloud compute tpus queued-resources delete "$name" --zone="$z" \
          --project="$PROJECT" --force --quiet >/dev/null 2>&1
      done
      sleep 20
      timeout 300 gcloud compute tpus queued-resources list --zone="$z" --project="$PROJECT" \
        --format='value(name,state.state)' 2>/dev/null | grep -E "^${QR_PREFIX}-" \
        && echo "[cleanup] STILL PRESENT in $z -- MANUAL ACTION NEEDED" \
        || echo "[cleanup] verified empty in $z"
    else
      echo "[cleanup] verified empty in $z"
    fi
  done
}
trap cleanup EXIT
trap 'echo "[trap] caught TERM/INT/HUP"; cleanup; exit 143' TERM INT HUP

# ------------------------------------------------------------------- queue
echo "=== phase 5: queue on $(hostname):${PORT} $(date +%H:%M:%S) ==="
uv run --isolated --extra dev --extra jax --with fastapi --with uvicorn --with flatbuffers \
  python -m pallas_arena.judge.queue --port "$PORT" --host 127.0.0.1 --lease-timeout "$LEASE_TIMEOUT_S" \
  > "${out_dir}/phase5-queue-${SLURM_JOB_ID}.log" 2>&1 &
queue_pid=$!
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${PORT}/status" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/status" >/dev/null || { echo "queue failed to start"; exit 1; }

# ----------------------------------------------------------------- create N
echo "=== creating ${N} spot ${ACCEL} QRs in ${ZONE} $(date +%H:%M:%S) ==="
for name in "${judges[@]}"; do
  gc queued-resources create "$name" --node-id="$name" --accelerator-type="$ACCEL" \
     --runtime-version="$RUNTIME" --spot 2>&1 | tail -1 &
done
wait
t_first_qr=$(date +%s)

( sleep "$HARD_CAP_S"
  echo "[watchdog] $(date +%H:%M:%S) HARD CAP ${HARD_CAP_S}s since first QR create -- tearing down"
  kill -TERM $$ ) &
watchdog_pid=$!

# ------------------------------------------------------- per-slot supervisor
# Each slot independently: wait ACTIVE -> provision -> hold the worker up
# through a reverse tunnel -> repeat if the ssh session (or the worker) dies.
# Being per-slot and restart-driven is what lets the compile-budget watchdog
# hard-exit a judge (and the chaos test SIGKILL one) without ending the demo.
slot_supervisor() {
  local name=$1
  local log="${out_dir}/phase5-judge-${name}-${SLURM_JOB_ID}.log"
  while true; do
    local st=""
    local deadline=$(( $(date +%s) + LAND_DEADLINE_S ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
      st=$(gc queued-resources describe "$name" --format='value(state.state)' 2>/dev/null)
      [ "$st" = "ACTIVE" ] && break
      case "$st" in FAILED|SUSPENDED|SUSPENDING|"") echo "[$name] state=${st:-gone}" >> "$log" ;; esac
      sleep 30
    done
    if [ "$st" != "ACTIVE" ]; then
      echo "[$name] not ACTIVE (state=${st:-gone}); retrying" >> "$log"
      sleep 60
      continue
    fi
    echo "[$name] ACTIVE at $(date +%H:%M:%S); provisioning" >> "$log"
    for _ in 1 2 3 4 5 6; do tssh "$name" "echo ssh-ok" >> "$log" 2>&1 && break; sleep 20; done
    timeout 900 gcloud compute tpus tpu-vm scp --recurse \
      "${repo_root}/tpu/pallas_arena" "${name}:~/arena-pallas_arena" \
      --zone="$ZONE" --project="$PROJECT" --worker=0 >> "$log" 2>&1
    tssh "$name" "rm -rf ~/arena && mkdir -p ~/arena && mv ~/arena-pallas_arena ~/arena/pallas_arena" >> "$log" 2>&1
    if ! tssh "$name" "ARENA_CACHE_PREFIX='${CACHE}' bash ~/arena/pallas_arena/phase2/provision_judge.sh" >> "$log" 2>&1; then
      echo "[$name] provision failed; retrying" >> "$log"
      sleep 60
      continue
    fi
    touch "${ready_dir}/${name}"
    echo "[$name] provisioned at $(date +%H:%M:%S); starting worker loop" >> "$log"
    while true; do
      timeout 14400 gcloud compute tpus tpu-vm ssh "$name" --zone="$ZONE" --project="$PROJECT" --worker=0 \
        --command="cd ~/arena && ARENA_CHILD_JAX_PLATFORMS=cpu ARENA_RLIMIT_GB=32 PYTHONPATH=~/arena \
          ~/arena-venv/bin/python -m pallas_arena.judge.worker \
          --problem rmsnorm --queue http://127.0.0.1:${PORT} --sim-mode real \
          --timing-pairs 20 --worker-id ${name} --compile-budget-s ${COMPILE_BUDGET_S} \
          --cache '${CACHE}' --boot-report \$HOME/boot-report.json --poll-s 2 \
          2>&1 | tee -a \$HOME/worker-progress.log" \
        -- -R "${PORT}:localhost:${PORT}" -o ExitOnForwardFailure=yes \
           -o ServerAliveInterval=30 -o ServerAliveCountMax=4 >> "$log" 2>&1
      echo "[$name] worker ssh session ended rc=$? $(date +%H:%M:%S); restarting in 10s" >> "$log"
      sleep 10
    done
  done
}

for name in "${judges[@]}"; do
  slot_supervisor "$name" &
  sup_pids+=($!)
done

# -------------------------------------------------------- wait for landings
echo "=== waiting up to ${LAND_DEADLINE_S}s for ${N} judges $(date +%H:%M:%S) ==="
land_deadline=$(( t_first_qr + LAND_DEADLINE_S ))
while [ "$(date +%s)" -lt "$land_deadline" ]; do
  ready=$(ls "$ready_dir" 2>/dev/null | wc -l)
  echo "  ready ${ready}/${N} $(date +%H:%M:%S)"
  [ "$ready" -ge "$N" ] && break
  sleep 30
done
landed=$(ls "$ready_dir" 2>/dev/null | sort | tr '\n' ',' | sed 's/,$//')
n_landed=$(ls "$ready_dir" 2>/dev/null | wc -l)
echo "=== ${n_landed}/${N} judges landed: ${landed:-none} $(date +%H:%M:%S) ==="

if [ "$n_landed" -lt "$N" ]; then
  echo "=== degraded fleet: deleting the QRs that never landed ==="
  for name in "${judges[@]}"; do
    if [ ! -e "${ready_dir}/${name}" ]; then
      echo "  deleting pending ${name}"
      gc queued-resources delete "$name" --force --quiet >/dev/null 2>&1 &
    fi
  done
  wait
fi
if [ "$n_landed" -lt "$MIN_JUDGES" ]; then
  echo "FATAL: only ${n_landed} judges; need >= ${MIN_JUDGES}"
  exit 1
fi

# ------------------------------------------------------------------- keeper
# Replacement is only enabled when the full fleet landed: with all N slots
# accounted for, `delete-then-create into the slot we just freed` cannot
# breach the ceiling. On a degraded fleet the keeper stays in delete-dead
# mode -- never create a 6th to retry a slow one.
keeper_replace=""
[ "$n_landed" -ge "$N" ] && keeper_replace="--replace"
uv run --isolated --extra dev --extra jax python -m pallas_arena.phase5.keeper \
  --prefix "$QR_PREFIX" --size "$N" --zone "$ZONE" --project "$PROJECT" \
  --accel "$ACCEL" --runtime "$RUNTIME" --interval 120 --max-replacements 2 \
  --deadline-s $(( HARD_CAP_S - 900 )) --state-file "${out_dir}/phase5-keeper-${SLURM_JOB_ID}.json" \
  $keeper_replace > "${out_dir}/phase5-keeper-${SLURM_JOB_ID}.log" 2>&1 &
keeper_pid=$!

# ------------------------------------------------------------------- driver
echo "=== running fleet driver $(date +%H:%M:%S) ==="
uv run --isolated --extra dev --extra jax --with fastapi --with uvicorn --with flatbuffers \
  python tpu/pallas_arena/phase5/fleet_driver.py \
  --queue "http://127.0.0.1:${PORT}" \
  --out "${out_dir}/phase5-results-${SLURM_JOB_ID}.json" \
  --judges "$landed" --judges-expected "$n_landed" --judges-min "$MIN_JUDGES" \
  --target "$TARGET" --dups "$DUPS" --expect-tpu \
  --zone "$ZONE" --project "$PROJECT" \
  --fleet-wait-s 900 --grade-wait-s 4800 --chaos-after 0.35
rc=$?

# ---------------------------------------------------------------- artifacts
echo "=== fetching artifacts $(date +%H:%M:%S) ==="
for p in "${sup_pids[@]:-}"; do kill "$p" 2>/dev/null; done
sup_pids=()
pkill -P $$ -f "gcloud compute tpus tpu-vm ssh" 2>/dev/null
for name in $(ls "$ready_dir" 2>/dev/null); do
  timeout 300 gcloud compute tpus tpu-vm scp "${name}:~/boot-report.json" \
    "${out_dir}/phase5-boot-${name}-${SLURM_JOB_ID}.json" \
    --zone="$ZONE" --project="$PROJECT" --worker=0 >/dev/null 2>&1 || true
  timeout 300 gcloud compute tpus tpu-vm scp "${name}:~/worker-progress.log" \
    "${out_dir}/phase5-progress-${name}-${SLURM_JOB_ID}.log" \
    --zone="$ZONE" --project="$PROJECT" --worker=0 >/dev/null 2>&1 || true
done
curl -sf "http://127.0.0.1:${PORT}/status" > "${out_dir}/phase5-queue-status-${SLURM_JOB_ID}.json" 2>/dev/null || true

chip_s=$(( $(date +%s) - t_first_qr ))
echo "=== phase 5 done rc=${rc} $(date +%H:%M:%S); fleet lifetime ${chip_s}s x ${n_landed} judges ==="
exit "$rc"
