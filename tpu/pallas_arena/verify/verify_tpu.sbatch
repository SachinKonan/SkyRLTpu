#!/bin/bash
# ONE v6e-1 spot QR, one job: adversarially verify the arena's first above-1.0
# candidate (rg_lru, 1.1941) and measure the task's real ceiling. No serving
# slice, no generation, no RL -- every model output this uses is already cached
# in runs/pallas_arena/seam-results-3651278.jsonl.
#
# HARD RESOURCE RULES, in the code rather than in a comment:
#   * exactly ONE QR, distinctly named (sk7524-arenaverify-judge), deleted on
#     EVERY exit path -- success, failure, exception, TERM/INT/HUP (slurm
#     timeout, scancel, node drain) and the hard cap;
#   * a 2 h hard cap measured from the QR create;
#   * DETACHED (`setsid nohup`) deletes issued FIRST so they outlive a scancel
#     that kills this shell 30 s later, then an in-band verification pass;
#   * the end state is verified empty in BOTH us-east5-a and us-east5-b.
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 2:45:00
#SBATCH --job-name=arena-verify-tpu
#SBATCH --signal=B:TERM@180
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/verify-tpu-%j.log

set -uo pipefail
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run under SLURM (sbatch); no compute on the login node." >&2
  exit 1
fi

PROJECT=${PROJECT:-vision-mix}
QR=${QR:-sk7524-arenaverify-judge}
ZONE=${ZONE:-us-east5-b}
OTHER_ZONE=${OTHER_ZONE:-us-east5-a}
ACCEL=${ACCEL:-v6e-1}
RUNTIME=${RUNTIME:-v2-alpha-tpuv6e}
LAND_DEADLINE_S=${LAND_DEADLINE_S:-2400}
HARD_CAP_S=${HARD_CAP_S:-7200}          # 2 h from QR create, per the rules
REPEATS=${REPEATS:-5}
TIMING_PAIRS=${TIMING_PAIRS:-20}

repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
out_dir="${repo_root}/runs/pallas_arena"
work_dir="${out_dir}/verify-work-${SLURM_JOB_ID}"
mkdir -p "$out_dir" "$work_dir"
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
  setsid nohup timeout 900 gcloud compute tpus queued-resources delete "$QR" \
    --zone="$ZONE" --project="$PROJECT" --force --quiet \
    > "${out_dir}/verify-del-${SLURM_JOB_ID}.log" 2>&1 < /dev/null &
  echo "[cleanup] detached delete issued for $QR"
  ( timeout 600 gcloud compute tpus queued-resources delete "$QR" --zone="$ZONE" \
      --project="$PROJECT" --force --quiet >/dev/null 2>&1; echo "[cleanup] delete confirmed: $QR" ) &
  del_pid=$!
  wait "$del_pid" 2>/dev/null

  for z in "$ZONE" "$OTHER_ZONE"; do
    left=$(timeout 300 gcloud compute tpus queued-resources list --zone="$z" --project="$PROJECT" \
             --format='value(name)' 2>/dev/null | grep -E "arenaverify" || true)
    if [ -n "$left" ]; then
      echo "[cleanup] SURVIVORS in $z: $left -- deleting again"
      for name in $left; do
        timeout 600 gcloud compute tpus queued-resources delete "$name" --zone="$z" \
          --project="$PROJECT" --force --quiet >/dev/null 2>&1
      done
      sleep 25
      timeout 300 gcloud compute tpus queued-resources list --zone="$z" --project="$PROJECT" \
        --format='value(name,state.state)' 2>/dev/null | grep -E "arenaverify" \
        && echo "[cleanup] QRs STILL PRESENT in $z -- MANUAL ACTION NEEDED" \
        || echo "[cleanup] verified empty of arenaverify QRs in $z"
    else
      echo "[cleanup] verified empty of arenaverify QRs in $z"
    fi
    nodes=$(timeout 300 gcloud compute tpus tpu-vm list --zone="$z" --project="$PROJECT" \
              --format='value(name)' 2>/dev/null | grep -E "arenaverify" || true)
    if [ -n "$nodes" ]; then
      echo "[cleanup] ORPHAN NODES in $z: $nodes -- deleting"
      for n in $nodes; do
        timeout 600 gcloud compute tpus tpu-vm delete "$n" --zone="$z" --project="$PROJECT" --quiet >/dev/null 2>&1
      done
    else
      echo "[cleanup] verified empty of arenaverify nodes in $z"
    fi
  done
  [ -n "$t_qr" ] && echo "[cleanup] QR lifetime $(( $(date +%s) - t_qr ))s"
}
trap cleanup EXIT
trap 'echo "[trap] caught TERM/INT/HUP"; cleanup; exit 143' TERM INT HUP

# ------------------------------------------------- the candidate codes (CPU)
codes="${work_dir}/codes.json"
uv run --isolated --extra dev --extra jax python - "$codes" <<'PY'
import json, sys
rows = [json.loads(l) for l in open("runs/pallas_arena/seam-results-3651278.jsonl")]
sel = {}
for r in rows:
    if r["task"] != "rg_lru" or not (r.get("reward") or 0) > 0:
        continue
    # the four above-1.0 candidates, plus one exactly-1.0 (a verbatim
    # re-implementation of the baseline) as an internal control
    if (r["reward"] > 1.0 and r["variant"] == "seam") or abs(r["reward"] - 1.0) < 1e-9:
        sel[f"{r['model']}-{r['variant']}-i{r['idx']}-r{r['reward']:.4f}"] = r["code"]
json.dump(sel, open(sys.argv[1], "w"), indent=1)
print("codes:", list(sel))
PY
[ -s "$codes" ] || { echo "FATAL: no candidate codes extracted"; exit 1; }

# ------------------------------------------------------------------- the QR
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
timeout 300 gcloud compute tpus tpu-vm scp "$codes" "${QR}:~/codes.json" \
  --zone="$ZONE" --project="$PROJECT" --worker=0
$SSH --command="bash ~/arena/pallas_arena/phase2/provision_judge.sh" || {
  echo "FATAL: provision failed"; exit 1; }

echo "=== [$(date +%H:%M:%S)] running tpu_verify on chip ==="
timeout 6000 gcloud compute tpus tpu-vm ssh "$QR" --zone="$ZONE" --project="$PROJECT" --worker=0 \
  --command="cd ~/arena && ARENA_CHILD_JAX_PLATFORMS=cpu ARENA_RLIMIT_GB=48 PYTHONPATH=~/arena \
    ~/arena-venv/bin/python -m pallas_arena.verify.tpu_verify \
      --codes ~/codes.json --out ~/verify-tpu.json \
      --repeats ${REPEATS} --timing-pairs ${TIMING_PAIRS} 2>&1 | tee \$HOME/verify-tpu.log"
rc=$?
echo "=== [$(date +%H:%M:%S)] on-chip run rc=${rc} ==="

timeout 300 gcloud compute tpus tpu-vm scp "${QR}:~/verify-tpu.json" \
  "${out_dir}/verify-tpu-${SLURM_JOB_ID}.json" --zone="$ZONE" --project="$PROJECT" --worker=0 || true
timeout 300 gcloud compute tpus tpu-vm scp "${QR}:~/verify-tpu.log" \
  "${out_dir}/verify-tpu-chip-${SLURM_JOB_ID}.log" --zone="$ZONE" --project="$PROJECT" --worker=0 || true

echo "=== done $(date +%H:%M:%S); tearing down ==="
exit "$rc"
