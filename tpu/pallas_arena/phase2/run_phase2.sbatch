#!/bin/bash
# Phase-2 orchestrator: ONE spot v6e-1 QR in us-east5-b -> provision ->
# RMSNorm shakedown -> fetch results -> ALWAYS delete the QR.
#
# HARD RULES ENCODED HERE (mandate):
#   * at most one QR, fixed name sk7524-pallas-judge-shakedown; this script
#     never touches any other QR/TPU (the running sweep is off-limits);
#   * if the QR isn't ACTIVE within WAIT_ACTIVE_S (2h), delete it and bail;
#   * the QR is deleted on EVERY exit path (trap), success or failure.
#
# Runs under sbatch on a neuronic compute node (gcloud + ssh only; the
# actual compute happens on the TPU host).
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH -t 4:00:00
#SBATCH --job-name=pallas-arena-phase2
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/phase2-%j.log

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
WAIT_ACTIVE_S=${WAIT_ACTIVE_S:-7200}
SHAKEDOWN_WAIT_S=${SHAKEDOWN_WAIT_S:-4800}
repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
out_dir="${repo_root}/runs/pallas_arena"
mkdir -p "$out_dir"

gc() { gcloud compute tpus "$@" --zone="$ZONE" --project="$PROJECT"; }
tssh() { timeout 300 gcloud compute tpus tpu-vm ssh "$QR_NAME" --zone="$ZONE" \
           --project="$PROJECT" --worker=0 --command="$1"; }

cleanup() {
  echo "[cleanup] deleting QR ${QR_NAME} (always-delete rule) $(date +%H:%M:%S)"
  gc queued-resources delete "$QR_NAME" --force --quiet 2>&1 | tail -2 || true
}
trap cleanup EXIT

echo "=== phase 2: creating ONE spot ${ACCEL} QR in ${ZONE} $(date +%H:%M:%S) ==="
if ! gc queued-resources create "$QR_NAME" \
    --node-id="$QR_NAME" \
    --accelerator-type="$ACCEL" \
    --runtime-version="$RUNTIME" \
    --spot 2>&1 | tail -3; then
  echo "QR create failed"; exit 1
fi

deadline=$(( $(date +%s) + WAIT_ACTIVE_S ))
state=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  state=$(gc queued-resources describe "$QR_NAME" --format='value(state.state)' 2>/dev/null)
  echo "  qr state: ${state:-unknown} $(date +%H:%M:%S)"
  case "$state" in
    ACTIVE) break ;;
    FAILED|SUSPENDED|SUSPENDING)
      echo "QR entered terminal state ${state}; bailing"; exit 1 ;;
  esac
  sleep 60
done
if [ "$state" != "ACTIVE" ]; then
  echo "QR not ACTIVE within ${WAIT_ACTIVE_S}s (floundering rule) — deleting"
  exit 1
fi
echo "=== QR ACTIVE $(date +%H:%M:%S); provisioning ==="

# ssh can need a retry while the VM finishes booting
for i in 1 2 3 4 5; do
  tssh "echo ssh-ok" && break
  echo "  ssh not ready (try $i); sleeping 30s"; sleep 30
done

timeout 300 gcloud compute tpus tpu-vm scp --recurse \
  "${repo_root}/tpu/pallas_arena" "${QR_NAME}:~/arena-pallas_arena" \
  --zone="$ZONE" --project="$PROJECT" --worker=0
tssh "rm -rf ~/arena && mkdir -p ~/arena && mv ~/arena-pallas_arena ~/arena/pallas_arena"
tssh "bash ~/arena/pallas_arena/phase2/provision_judge.sh" || { echo "provision failed"; exit 1; }

echo "=== launching shakedown in tmux $(date +%H:%M:%S) ==="
tssh "tmux kill-session -t shakedown 2>/dev/null; tmux new-session -d -s shakedown \
  'cd ~/arena && JAX_PLATFORMS=cpu ARENA_CHILD_JAX_PLATFORMS=tpu ARENA_RLIMIT_GB=512 \
   PYTHONPATH=~/arena \
   ~/arena-venv/bin/python pallas_arena/phase2/shakedown.py \
   --out ~/shakedown-results.json > ~/shakedown.log 2>&1'"

deadline=$(( $(date +%s) + SHAKEDOWN_WAIT_S ))
done=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  if tssh "test -f ~/shakedown-results.json && echo DONE" 2>/dev/null | grep -q DONE; then
    done=yes; break
  fi
  echo "  shakedown running... $(date +%H:%M:%S)"
  sleep 120
done

echo "=== fetching artifacts $(date +%H:%M:%S) ==="
timeout 300 gcloud compute tpus tpu-vm scp \
  "${QR_NAME}:~/shakedown.log" "${out_dir}/phase2-shakedown-${SLURM_JOB_ID}.log" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 || true
timeout 300 gcloud compute tpus tpu-vm scp \
  "${QR_NAME}:~/shakedown-results.json" "${out_dir}/phase2-shakedown-results-${SLURM_JOB_ID}.json" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 || true
timeout 300 gcloud compute tpus tpu-vm scp \
  "${QR_NAME}:~/judge-server.log" "${out_dir}/phase2-judge-server-${SLURM_JOB_ID}.log" \
  --zone="$ZONE" --project="$PROJECT" --worker=0 || true

if [ -z "$done" ]; then
  echo "shakedown did not finish within ${SHAKEDOWN_WAIT_S}s"; exit 1
fi
echo "=== phase 2 done $(date +%H:%M:%S) (QR deleted by trap) ==="
exit 0
