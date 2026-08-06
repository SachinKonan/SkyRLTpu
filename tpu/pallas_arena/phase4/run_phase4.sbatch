#!/bin/bash
# Phase-4 orchestrator: queue on THIS compute node; ONE spot v6e-1 worker
# polls it over a reverse ssh tunnel; RMSNorm acceptance battery must turn
# every phase-2 red green. Same hard rules as phase 2: one QR at a time
# (fixed name, never touches anything else), 2h floundering cap, QR deleted
# on EVERY exit path.
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 4:00:00
#SBATCH --job-name=pallas-arena-phase4
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/phase4-%j.log

set -uo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run under SLURM (sbatch)." >&2
  exit 1
fi

QR_NAME=sk7524-pallas-judge-shakedown
ZONE=us-east5-b
PROJECT=vision-mix
ACCEL=v6e-1
RUNTIME=v2-alpha-tpuv6e
PORT=${PORT:-8770}
WAIT_ACTIVE_S=${WAIT_ACTIVE_S:-7200}
repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
out_dir="${repo_root}/runs/pallas_arena"
mkdir -p "$out_dir"
cd "$repo_root"
export PYTHONPATH="${repo_root}/tpu:${PYTHONPATH:-}"   # pallas_arena.* import root

gc() { gcloud compute tpus "$@" --zone="$ZONE" --project="$PROJECT"; }
tssh() { timeout 300 gcloud compute tpus tpu-vm ssh "$QR_NAME" --zone="$ZONE" \
           --project="$PROJECT" --worker=0 --command="$1"; }

queue_pid=""
worker_pid=""
cleanup() {
  echo "[cleanup] $(date +%H:%M:%S) killing worker/queue, deleting QR (always-delete)"
  [ -n "$worker_pid" ] && kill "$worker_pid" 2>/dev/null
  [ -n "$queue_pid" ] && kill "$queue_pid" 2>/dev/null
  gc queued-resources delete "$QR_NAME" --force --quiet 2>&1 | tail -2 || true
}
trap cleanup EXIT

echo "=== phase 4: starting queue on $(hostname):${PORT} $(date +%H:%M:%S) ==="
uv run --isolated --extra dev --extra jax --with fastapi --with uvicorn --with flatbuffers \
  python -m pallas_arena.judge.queue --port "$PORT" --host 127.0.0.1 --lease-timeout 240 \
  > "${out_dir}/phase4-queue-${SLURM_JOB_ID}.log" 2>&1 &
queue_pid=$!
for i in $(seq 1 60); do
  curl -sf "http://127.0.0.1:${PORT}/status" >/dev/null 2>&1 && break
  sleep 2
done
curl -sf "http://127.0.0.1:${PORT}/status" >/dev/null || { echo "queue failed to start"; exit 1; }

echo "=== creating ONE spot ${ACCEL} QR $(date +%H:%M:%S) ==="
gc queued-resources create "$QR_NAME" \
  --node-id="$QR_NAME" --accelerator-type="$ACCEL" \
  --runtime-version="$RUNTIME" --spot 2>&1 | tail -2 || { echo "QR create failed"; exit 1; }

deadline=$(( $(date +%s) + WAIT_ACTIVE_S ))
state=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  state=$(gc queued-resources describe "$QR_NAME" --format='value(state.state)' 2>/dev/null)
  echo "  qr state: ${state:-unknown} $(date +%H:%M:%S)"
  case "$state" in
    ACTIVE) break ;;
    FAILED|SUSPENDED|SUSPENDING) echo "QR terminal state ${state}"; exit 1 ;;
  esac
  sleep 60
done
[ "$state" = "ACTIVE" ] || { echo "QR not ACTIVE in ${WAIT_ACTIVE_S}s (floundering rule)"; exit 1; }

for i in 1 2 3 4 5; do tssh "echo ssh-ok" && break; sleep 30; done
timeout 300 gcloud compute tpus tpu-vm scp --recurse \
  "${repo_root}/tpu/pallas_arena" "${QR_NAME}:~/arena-pallas_arena" \
  --zone="$ZONE" --project="$PROJECT" --worker=0
tssh "rm -rf ~/arena && mkdir -p ~/arena && mv ~/arena-pallas_arena ~/arena/pallas_arena"
tssh "bash ~/arena/pallas_arena/phase2/provision_judge.sh" || { echo "provision failed"; exit 1; }

echo "=== launching worker over reverse tunnel $(date +%H:%M:%S) ==="
# The ssh session IS the tunnel: the worker on the TPU host reaches this
# node's queue at 127.0.0.1:${PORT} through -R. Restart loop survives ssh
# drops; the driver decides when we are done.
(
  while true; do
    timeout 3600 gcloud compute tpus tpu-vm ssh "$QR_NAME" --zone="$ZONE" --project="$PROJECT" --worker=0 \
      --command="cd ~/arena && ARENA_CHILD_JAX_PLATFORMS=cpu ARENA_RLIMIT_GB=32 PYTHONPATH=~/arena \
        ~/arena-venv/bin/python -m pallas_arena.judge.worker \
        --problem rmsnorm --queue http://127.0.0.1:${PORT} --sim-mode real \
        --timing-pairs 20 --worker-id v6e-judge --boot-report /home/sk7524/boot-report.json --poll-s 2" \
      -- -R "${PORT}:localhost:${PORT}" -o ServerAliveInterval=30 -o ServerAliveCountMax=4 \
      >> "${out_dir}/phase4-worker-${SLURM_JOB_ID}.log" 2>&1
    echo "[worker-ssh] session ended rc=$? $(date +%H:%M:%S); restarting in 15s" \
      >> "${out_dir}/phase4-worker-${SLURM_JOB_ID}.log"
    sleep 15
  done
) &
worker_pid=$!

echo "=== running driver $(date +%H:%M:%S) ==="
uv run --isolated --extra dev --extra jax --with fastapi --with uvicorn --with flatbuffers \
  python tpu/pallas_arena/phase4/driver.py \
  --queue "http://127.0.0.1:${PORT}" \
  --out "${out_dir}/phase4-results-${SLURM_JOB_ID}.json" \
  --worker-wait-s 2400 --grade-wait-s 3600
rc=$?

echo "=== fetching artifacts $(date +%H:%M:%S) ==="
kill "$worker_pid" 2>/dev/null; worker_pid=""
timeout 300 gcloud compute tpus tpu-vm scp \
  "${QR_NAME}:~/boot-report.json" "${out_dir}/phase4-boot-report-${SLURM_JOB_ID}.json" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 || true

echo "=== phase 4 done rc=${rc} $(date +%H:%M:%S) (QR deleted by trap) ==="
exit "$rc"
