#!/bin/bash
# RL GRADING FLEET for the pallas kernel arena (derived from phase5/fleet.sbatch,
# which proved this exact shape at fleet scale: 1699 items, 0 lost/dup).
#
# Differences from phase 5: PERSISTENT (no corpus driver -- the ttt-discover RL
# client feeds the queue and reads verdicts), the queue binds 0.0.0.0 so the
# client's slurm node can reach it across the cluster network (judges still use
# reverse tunnels), problem = rg_lru, and the queue URL is published to
# runs/pallas_arena/rl-queue-url.txt for the client sbatch to read.
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
#SBATCH -t 24:00:00
#SBATCH --job-name=pallas-rl-judges
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/rl-judges-%j.log

set -uo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run under SLURM (sbatch); no compute on the login node." >&2
  exit 1
fi

N=${N:-2}
QR_PREFIX=${QR_PREFIX:-sk7524-rl-judge}
ZONE=${ZONE:-us-east5-a}
PROJECT=${PROJECT:-vision-mix}
ACCEL=${ACCEL:-v5p-8}
RUNTIME=${RUNTIME:-v2-alpha-tpuv5}
PORT=${PORT:-8791}
LAND_DEADLINE_S=${LAND_DEADLINE_S:-14400}    # 4h: spot droughts have run hours (measured 2026-08-24/25)
HARD_CAP_S=${HARD_CAP_S:-84600}              # 23.5h; slurm -t 24h is the outer bound
LEASE_TIMEOUT_S=${LEASE_TIMEOUT_S:-240}
COMPILE_BUDGET_S=${COMPILE_BUDGET_S:-90}
# FRESH prefix: the swept-backward contract (e76a5ee4) changed every reward;
# a cache from the old fold must never serve RL verdicts.
CACHE=${CACHE:-gs://sk7524-pallas-arena-us-east5/reward-cache-rl-rglru-v1}
# Comma list; judge i grades problem PROBLEMS[(i-1) % len] -- lets one fleet
# carry rg_lru AND splash judges (seed-parity validation needs both).
PROBLEMS=${PROBLEMS:-rg_lru}
IFS=',' read -r -a problems_arr <<< "$PROBLEMS"
problem_for() { local idx="${1##*-}"; echo "${problems_arr[$(( (idx-1) % ${#problems_arr[@]} ))]}"; }
MIN_JUDGES=${MIN_JUDGES:-2}

repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
out_dir="${repo_root}/runs/pallas_arena"
work_dir="${out_dir}/rl-judges-work-${SLURM_JOB_ID}"
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
  # the supervisors' gcloud-ssh grandchildren hold the reverse tunnels and
  # would outlive their parents; the pattern is scoped to our own QR names
  pkill -P $$ 2>/dev/null
  pkill -f "tpu-vm ssh ${QR_PREFIX}-" 2>/dev/null
  sleep 1
  del_pids=()
  for name in "${judges[@]}"; do
    ( timeout 600 gcloud compute tpus queued-resources delete "$name" \
        --zone="$ZONE" --project="$PROJECT" --force --quiet >/dev/null 2>&1
      echo "[cleanup] delete issued: $name" ) &
    del_pids+=($!)
  done
  # NEVER a bare `wait` in this script: the queue server is a background child
  # that by design never exits, so `wait` with no arguments blocks forever.
  # (Measured: run 3648750 wedged here for 4 minutes with 5 QRs alive before
  # it was killed.) Always wait on the explicit PIDs we care about.
  wait "${del_pids[@]}" 2>/dev/null
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
        && echo "[cleanup] QRs STILL PRESENT in $z -- MANUAL ACTION NEEDED" \
        || echo "[cleanup] verified empty of arena QRs in $z"
    else
      echo "[cleanup] verified empty of arena QRs in $z"
    fi
    # A QR delete takes its node with it, but "should" is not "did": phase 2
    # made checking both the standing rule, so check both.
    nodes=$(timeout 300 gcloud compute tpus tpu-vm list --zone="$z" --project="$PROJECT" \
              --format='value(name)' 2>/dev/null | grep -E "^${QR_PREFIX}-" || true)
    if [ -n "$nodes" ]; then
      echo "[cleanup] ORPHAN NODES in $z: $nodes -- deleting"
      for n in $nodes; do
        timeout 600 gcloud compute tpus tpu-vm delete "$n" --zone="$z" --project="$PROJECT" --quiet >/dev/null 2>&1
      done
    else
      echo "[cleanup] verified empty of arena nodes in $z"
    fi
  done
}
trap cleanup EXIT
trap 'echo "[trap] caught TERM/INT/HUP"; cleanup; exit 143' TERM INT HUP

# ------------------------------------------------------------------- queue
echo "=== phase 5: queue on $(hostname):${PORT} $(date +%H:%M:%S) ==="
uv run --isolated --extra dev --extra jax --with fastapi --with uvicorn --with flatbuffers \
  python -m pallas_arena.judge.queue --port "$PORT" --host 0.0.0.0 --lease-timeout "$LEASE_TIMEOUT_S" \
  > "${out_dir}/rl-queue-${SLURM_JOB_ID}.log" 2>&1 &
queue_pid=$!
for _ in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${PORT}/status" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/status" >/dev/null || { echo "queue failed to start"; exit 1; }

# ----------------------------------------------------------------- create N
echo "=== creating ${N} spot ${ACCEL} QRs in ${ZONE} $(date +%H:%M:%S) ==="
create_pids=()
for name in "${judges[@]}"; do
  gc queued-resources create "$name" --node-id="$name" --accelerator-type="$ACCEL" \
     --runtime-version="$RUNTIME" --spot 2>&1 | tail -1 &
  create_pids+=($!)
done
wait "${create_pids[@]}" 2>/dev/null
t_first_qr=$(date +%s)

( trap - EXIT TERM INT HUP
  sleep "$HARD_CAP_S"
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
  # bash resets traps in a subshell, so this function does NOT inherit the
  # EXIT/TERM cleanup — verified, and relied upon: a supervisor exiting must
  # never tear down the fleet. Made explicit so a future edit cannot make it
  # so by accident.
  trap - EXIT TERM INT HUP
  while true; do
    local st=""
    local deadline=$(( $(date +%s) + LAND_DEADLINE_S ))
    while [ "$(date +%s)" -lt "$deadline" ]; do
      st=$(gc queued-resources describe "$name" --format='value(state.state)' 2>/dev/null)
      [ "$st" = "ACTIVE" ] && break
      # A GCP queued resource in FAILED/SUSPENDED is TERMINAL -- it never
      # returns to the queue on its own. Measured 2026-08-25: a v6e judge
      # FAILED with "no more capacity in the zone" 90s after create and then
      # sat dead for the whole landing window while this loop polled it.
      # Delete and recreate so the request is back in line for capacity.
      case "$st" in
        FAILED|SUSPENDED|SUSPENDING|"")
          echo "[$name] state=${st:-gone} -- deleting and recreating $(date +%H:%M:%S)" >> "$log"
          gc queued-resources delete "$name" --force --quiet >/dev/null 2>&1
          sleep 30
          gc queued-resources create "$name" --node-id="$name" \
            --accelerator-type="$ACCEL" --runtime-version="$RUNTIME" --spot >> "$log" 2>&1
          sleep 60
          ;;
      esac
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
    # Restart loop. It has to survive three different deaths that all look
    # alike from here: an ssh drop, the chaos test's SIGKILL, and the
    # compile-budget watchdog's deliberate os._exit(17) after it has already
    # recorded the terminal verdict. What it must NOT do is spin forever
    # against a preempted host, so three consecutive short sessions send us
    # back to the QR state check (where the keeper may have replaced it).
    local fails=0 t_s dur rc
    while true; do
      t_s=$(date +%s)
      if [ "${GRADER:-ray}" = "ray" ]; then
        # RAY POOL: a v5p-8 judge host has 4 chips and the single-worker path
        # uses one. One actor per chip = 4 concurrent grades per host.
        remote_cmd="set -o pipefail; cd ~/arena && \
          RAY_CHIPS=${RAY_CHIPS:-4} bash ~/arena/pallas_arena/phase2/ray_start_tpu.sh; \
          PYTHONPATH=~/arena JAX_COMPILATION_CACHE_DIR=\$HOME/jax-compile-cache \
          ~/arena-venv/bin/python -m pallas_arena.judge.ray_pool \
          --queue http://127.0.0.1:${PORT} --problems ${PROBLEMS} \
          --actors ${RAY_ACTORS:-4} --width ${RAY_WIDTH:-1} \
          --timing-pairs 20 --compile-budget-s ${COMPILE_BUDGET_S} \
          --cache '${CACHE}' --poll-s 1 ${IDLE_EXIT_S:+--idle-exit-s ${IDLE_EXIT_S}} \
          2>&1 | tee -a \$HOME/worker-progress.log"
      else
        remote_cmd="set -o pipefail; cd ~/arena && ARENA_CHILD_JAX_PLATFORMS=cpu ARENA_RLIMIT_GB=32 PYTHONPATH=~/arena \
          ~/arena-venv/bin/python -m pallas_arena.judge.worker \
          --problem $(problem_for "$name") --queue http://127.0.0.1:${PORT} --sim-mode real \
          --timing-pairs 20 --worker-id ${name} --compile-budget-s ${COMPILE_BUDGET_S} \
          --cache '${CACHE}' --boot-report \$HOME/boot-report.json --poll-s 2 \
          2>&1 | tee -a \$HOME/worker-progress.log"
      fi
      timeout 14400 gcloud compute tpus tpu-vm ssh "$name" --zone="$ZONE" --project="$PROJECT" --worker=0 \
        --command="$remote_cmd" \
        -- -R "${PORT}:localhost:${PORT}" -o ExitOnForwardFailure=yes \
           -o ServerAliveInterval=30 -o ServerAliveCountMax=4 >> "$log" 2>&1
      rc=$?
      dur=$(( $(date +%s) - t_s ))
      # `set -o pipefail` above is load-bearing: without it the remote `| tee`
      # swallows the worker's status and every death reports rc=0, which is
      # how a judge-fault cascade stays invisible in this log. 17 = compile
      # budget watchdog (terminal verdict already posted), 18 = judge fault
      # (chip dead, item deliberately left to requeue).
      case "$rc" in
        17) echo "[$name] worker exited on the COMPILE-BUDGET watchdog" >> "$log" ;;
        18) echo "[$name] worker exited on a JUDGE FAULT (dead chip); item requeued" >> "$log" ;;
      esac
      echo "[$name] worker ssh session ended rc=${rc} after ${dur}s $(date +%H:%M:%S)" >> "$log"
      if [ "$dur" -lt 60 ]; then fails=$(( fails + 1 )); else fails=0; fi
      if [ "$fails" -ge 3 ]; then
        echo "[$name] 3 consecutive short sessions; re-checking the QR" >> "$log"
        rm -f "${ready_dir}/${name}"
        break
      fi
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
  pend_pids=()
  for name in "${judges[@]}"; do
    if [ ! -e "${ready_dir}/${name}" ]; then
      echo "  deleting pending ${name}"
      gc queued-resources delete "$name" --force --quiet >/dev/null 2>&1 &
      pend_pids+=($!)
    fi
  done
  [ ${#pend_pids[@]} -gt 0 ] && wait "${pend_pids[@]}" 2>/dev/null
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
  --deadline-s $(( HARD_CAP_S - 900 )) --state-file "${out_dir}/rl-keeper-${SLURM_JOB_ID}.json" \
  $keeper_replace > "${out_dir}/rl-keeper-${SLURM_JOB_ID}.log" 2>&1 &
keeper_pid=$!

# ------------------------------------------------------------- serve forever
# Publish where the queue lives; the RL client sbatch reads this file.
echo "http://$(hostname -s):${PORT}" > "${out_dir}/rl-queue-url.txt"
echo "=== RL grading fleet up: $(cat "${out_dir}/rl-queue-url.txt") (${n_landed} judges) ==="

# Hold the fleet up until the hard cap; report queue status every 5 min.
serve_end=$(( t_first_qr + HARD_CAP_S ))
while [ "$(date +%s)" -lt "$serve_end" ]; do
  sleep 300
  st=$(curl -sf -m 8 "http://127.0.0.1:${PORT}/status" 2>/dev/null || echo '{"unreachable":true}')
  echo "[status] $(date +%H:%M:%S) ${st}"
done
echo "=== hard cap reached; tearing down ==="
