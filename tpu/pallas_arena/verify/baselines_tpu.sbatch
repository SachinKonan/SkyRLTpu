#!/bin/bash
# ONE v5p-8 spot QR, one job: settle the arena's DENOMINATORS on silicon.
# No serving slice, no generation, no RL, no candidate code -- only the arena's
# own references/baselines/honest-variants plus tokamax's production kernels.
#
# Answers three things CPU cannot:
#   A. the recalibrated tolerance band on a real MXU. `preferred_element_type`
#      is a no-op inside a fused op on CPU, so megablox's CPU probe was an
#      artifact (sloppy read 0.67x, faithful read exactly 0.0) and its band is
#      formally unverified. splash/RPA/flce/rg_lru already pass on CPU;
#   B. WHICH baseline each task actually gets on this host -- four of five fall
#      back to a labelled non-production denominator, and a score against
#      `lax.ragged_dot` must never be reported as beating megablox;
#   C. whether tokamax (openxla, pins jax>=0.9.2 -> resolves to our own 0.10.2)
#      can replace those fallbacks: ragged_dot/mosaic for megablox,
#      dot_product_attention/mosaic for splash, and
#      linear_softmax_cross_entropy_loss/mosaic_tpu for flce as a TIMING-only
#      comparison (its public API has no reduction='none', so it returns a
#      scalar where our contract returns per-token logprobs).
#
# HARD RESOURCE RULES, in the code rather than in a comment:
#   * exactly ONE QR, distinctly named (sk7524-arenabase-judge), deleted on
#     EVERY exit path -- success, failure, exception, TERM/INT/HUP (slurm
#     timeout, scancel, node drain) and the hard cap;
#   * a 2 h hard cap measured from the QR create;
#   * DETACHED (`setsid nohup`) deletes issued FIRST so they outlive a scancel
#     that kills this shell ~30 s later, then an in-band verification pass;
#   * the end state is verified empty in BOTH us-east5-a and us-east5-b.
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:45:00
#SBATCH --job-name=arena-baselines-tpu
#SBATCH --signal=B:TERM@180
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/baselines-tpu-%j.log

set -uo pipefail
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run under SLURM (sbatch); no compute on the login node." >&2
  exit 1
fi

PROJECT=${PROJECT:-vision-mix}
# Job-unique name, NOT a fixed one: a fixed name races the PREVIOUS attempt's
# detached teardown. Measured on job 3691912 -- its QR was created 3 minutes
# after job 3691879's cleanup issued a `setsid nohup` delete (timeout 900) for
# the same name, and that delete took the new QR with it (state=gone at the
# landing check). The detached delete is load-bearing (it must survive a
# scancel that kills the shell mid-teardown), so the name must change instead.
# The cleanup sweep greps the "arenabase" prefix in both zones, so job-suffixed
# names are still caught by the survivor check.
QR=${QR:-sk7524-arenabase-judge-${SLURM_JOB_ID}}
ZONE=${ZONE:-us-east5-a}
OTHER_ZONE=${OTHER_ZONE:-us-east5-b}
ACCEL=${ACCEL:-v5p-8}
RUNTIME=${RUNTIME:-v2-alpha-tpuv5}
LAND_DEADLINE_S=${LAND_DEADLINE_S:-2400}
HARD_CAP_S=${HARD_CAP_S:-7200}
TASKS=${TASKS:-splash_attention,megablox_gmm,ragged_paged_attention,flce,rg_lru}
TAG=arenabase

repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
out_dir="${repo_root}/runs/pallas_arena"
mkdir -p "$out_dir"
cd "$repo_root"
export PYTHONPATH="${repo_root}/tpu:${PYTHONPATH:-}"

watchdog_pid=""
t_qr=""

cleaned=""
cleanup() {
  [ -n "$cleaned" ] && return 0
  cleaned=1
  echo "[cleanup] $(date +%H:%M:%S) tearing down ${QR}"
  [ -n "${watchdog_pid:-}" ] && kill "$watchdog_pid" 2>/dev/null
  pkill -P $$ 2>/dev/null
  pkill -f "tpu-vm ssh ${QR}" 2>/dev/null
  sleep 1
  # detached FIRST: a scancel kills this shell ~30 s in while one
  # queued-resource delete takes ~90 s to return (measured, job 3650988,
  # where both QRs survived their own teardown).
  setsid nohup timeout 900 gcloud compute tpus queued-resources delete "$QR" \
    --zone="$ZONE" --project="$PROJECT" --force --quiet \
    > "${out_dir}/${TAG}-del-${SLURM_JOB_ID}.log" 2>&1 < /dev/null &
  echo "[cleanup] detached delete issued for $QR"
  ( timeout 600 gcloud compute tpus queued-resources delete "$QR" --zone="$ZONE" \
      --project="$PROJECT" --force --quiet >/dev/null 2>&1; echo "[cleanup] delete confirmed: $QR" ) &
  del_pid=$!
  wait "$del_pid" 2>/dev/null

  for z in "$ZONE" "$OTHER_ZONE"; do
    left=$(timeout 300 gcloud compute tpus queued-resources list --zone="$z" --project="$PROJECT" \
             --format='value(name)' 2>/dev/null | grep -E "${TAG}" || true)
    if [ -n "$left" ]; then
      echo "[cleanup] SURVIVORS in $z: $left -- deleting again"
      for name in $left; do
        timeout 600 gcloud compute tpus queued-resources delete "$name" --zone="$z" \
          --project="$PROJECT" --force --quiet >/dev/null 2>&1
      done
      sleep 25
      timeout 300 gcloud compute tpus queued-resources list --zone="$z" --project="$PROJECT" \
        --format='value(name,state.state)' 2>/dev/null | grep -E "${TAG}" \
        && echo "[cleanup] QRs STILL PRESENT in $z -- MANUAL ACTION NEEDED" \
        || echo "[cleanup] verified empty of ${TAG} QRs in $z"
    else
      echo "[cleanup] verified empty of ${TAG} QRs in $z"
    fi
    nodes=$(timeout 300 gcloud compute tpus tpu-vm list --zone="$z" --project="$PROJECT" \
              --format='value(name)' 2>/dev/null | grep -E "${TAG}" || true)
    if [ -n "$nodes" ]; then
      echo "[cleanup] ORPHAN NODES in $z: $nodes -- deleting"
      for n in $nodes; do
        timeout 600 gcloud compute tpus tpu-vm delete "$n" --zone="$z" --project="$PROJECT" --quiet >/dev/null 2>&1
      done
    else
      echo "[cleanup] verified empty of ${TAG} nodes in $z"
    fi
  done
  [ -n "$t_qr" ] && echo "[cleanup] QR lifetime $(( $(date +%s) - t_qr ))s"
}
trap cleanup EXIT
trap 'echo "[trap] caught TERM/INT/HUP"; cleanup; exit 143' TERM INT HUP

echo "=== creating 1 spot QR ${QR} (${ACCEL}, ${ZONE}) $(date +%H:%M:%S) ==="
timeout 600 gcloud compute tpus queued-resources create "$QR" --node-id="$QR" \
  --accelerator-type="$ACCEL" --runtime-version="$RUNTIME" --spot \
  --zone="$ZONE" --project="$PROJECT" 2>&1 | tail -2
t_qr=$(date +%s)
echo "=== QR created at $(date +%H:%M:%S); HARD CAP at $(date -d "@$(( t_qr + HARD_CAP_S ))" +%H:%M:%S) ==="

( trap - EXIT TERM INT HUP
  sleep "$HARD_CAP_S"
  echo "[watchdog] HARD CAP ${HARD_CAP_S}s since QR create -- tearing down"
  kill -TERM $$ ) &
watchdog_pid=$!

st=""; deadline=$(( t_qr + LAND_DEADLINE_S ))
while [ "$(date +%s)" -lt "$deadline" ]; do
  st=$(timeout 300 gcloud compute tpus queued-resources describe "$QR" --zone="$ZONE" \
         --project="$PROJECT" --format='value(state.state)' 2>/dev/null)
  [ "$st" = "ACTIVE" ] && break
  case "$st" in FAILED) echo "[$QR] state=FAILED"; exit 1 ;; esac
  sleep 25
done
[ "$st" = "ACTIVE" ] || { echo "FATAL: ${QR} not ACTIVE within ${LAND_DEADLINE_S}s (state=${st:-gone})"; exit 1; }
echo "[$QR] ACTIVE at $(date +%H:%M:%S)"

SSH="timeout 900 gcloud compute tpus tpu-vm ssh $QR --zone=$ZONE --project=$PROJECT --worker=0"
for _ in 1 2 3 4 5 6; do $SSH --command="echo ssh-ok" && break; sleep 20; done

timeout 900 gcloud compute tpus tpu-vm scp --recurse "${repo_root}/tpu/pallas_arena" \
  "${QR}:~/arena-pallas_arena" --zone="$ZONE" --project="$PROJECT" --worker=0
$SSH --command="rm -rf ~/arena && mkdir -p ~/arena && mv ~/arena-pallas_arena ~/arena/pallas_arena"
$SSH --command="bash ~/arena/pallas_arena/phase2/provision_judge.sh" || {
  echo "FATAL: provision failed"; exit 1; }

# tokamax on TOP of the arena pin, never instead of it. It declares jax>=0.9.2,
# so 0.10.2 satisfies it; assert afterwards that the pin did NOT move, because a
# silent jax upgrade would invalidate every timing this arena has ever recorded.
echo "=== installing tokamax (arena pin must survive) ==="
$SSH --command="~/arena-venv/bin/pip install --quiet tokamax && \
  ~/arena-venv/bin/python -c \"
import jax, tokamax
print('jax', jax.__version__, 'tokamax', getattr(tokamax,'__version__','?'))
assert jax.__version__ == '0.10.2', 'ARENA JAX PIN MOVED: ' + jax.__version__
\"" || { echo "FATAL: tokamax install broke the arena jax pin"; exit 1; }

# ENTRY lets this same provisioned judge run a different on-chip measurement
# without duplicating 120 lines of QR lifecycle. Default: the denominator probe.
ENTRY=${ENTRY:-pallas_arena.verify.tpu_baselines}
# NOTE the double quotes: single quotes inside a ${var:-default} suppress the
# nested ${TASKS} expansion, so the remote shell got a literal ${TASKS} (empty
# there) and argparse died on "--tasks: expected one argument" (job 3691879).
ENTRY_ARGS=${ENTRY_ARGS:-"--tasks ${TASKS}"}

# Optional payload file (e.g. a codes.json for regrade_winners); lands at
# ~/payload.json on the judge, referenced from ENTRY_ARGS.
if [ -n "${EXTRA_SCP:-}" ]; then
  timeout 300 gcloud compute tpus tpu-vm scp "$EXTRA_SCP" "${QR}:~/payload.json" \
    --zone="$ZONE" --project="$PROJECT" --worker=0 || { echo "FATAL: payload scp failed"; exit 1; }
fi

echo "=== [$(date +%H:%M:%S)] running ${ENTRY} on chip ==="
timeout 6000 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --command="set -o pipefail; cd ~/arena && ARENA_CHILD_JAX_PLATFORMS=cpu PYTHONPATH=~/arena \
    ~/arena-venv/bin/python -m ${ENTRY} \
      --out ~/baselines-tpu.json ${ENTRY_ARGS} 2>&1 | tee \$HOME/baselines-tpu.log"
rc=$?
echo "=== [$(date +%H:%M:%S)] on-chip run rc=${rc} ==="

timeout 300 gcloud compute tpus tpu-vm scp "${QR}:~/baselines-tpu.json" \
  "${out_dir}/baselines-tpu-${SLURM_JOB_ID}.json" --zone="$ZONE" --project="$PROJECT" --worker=0 || true
timeout 300 gcloud compute tpus tpu-vm scp "${QR}:~/baselines-tpu.log" \
  "${out_dir}/baselines-tpu-chip-${SLURM_JOB_ID}.log" --zone="$ZONE" --project="$PROJECT" --worker=0 || true

echo "=== done $(date +%H:%M:%S); tearing down ==="
exit "$rc"
