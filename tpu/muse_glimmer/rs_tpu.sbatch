#!/bin/bash
# Slurm wrapper for the Muse-Glimmer reasoning-strength run.
#
# `--signal=TERM@300` deliberately WITHOUT the `B:` prefix so the TERM reaches
# rs_tpu.sh itself and its own trap runs; `B:TERM` would signal only this
# wrapper and the QR would leak.
#
# rs_tpu.sh is re-invoked after a preemption (exit 12) or a landed-but-
# unreachable host (exit 13), because both leave generated rollouts on the
# shared filesystem and the next attempt resumes from them. It is NOT
# re-invoked after a gate failure (11 = no channels, 14 = too slow) or after
# the cumulative slice cap (9): those are decisions, not accidents.
#
#SBATCH --job-name=mg-rs2
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=11:00:00
#SBATCH --signal=TERM@300
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/rs2-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
export PATH=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin:$HOME/.local/bin:$PATH
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

echo "=== node $(hostname) job ${SLURM_JOB_ID:-manual} $(date -Is) ==="

QR="${QR:-sk7524-museglimmer-rs2}"
export QR
ALL_ZONES="us-east5-a us-east5-b us-east5-c"
backstop() {
  echo "=== wrapper backstop: ensuring ${QR} is gone in ${ALL_ZONES} ==="
  for z in $ALL_ZONES; do
    setsid nohup "$GC" alpha compute tpus queued-resources delete "$QR" \
      --project=vision-mix --zone="$z" --quiet --force \
      >>"${REPO}/runs/muse_glimmer/qr-delete.log" 2>&1 </dev/null &
    disown 2>/dev/null || true
  done
  sleep 30
}
trap 'backstop' EXIT TERM INT HUP

# One attempt at a time, never two QRs. MAX_TRIES bounds the retries; the
# cumulative 6 h slice cap inside rs_tpu.sh bounds the chips.
MAX_TRIES="${MAX_TRIES:-4}"
rc=0
for try in $(seq 1 "$MAX_TRIES"); do
  echo "=== attempt ${try}/${MAX_TRIES} $(date -Is) ==="
  bash "${REPO}/tpu/muse_glimmer/rs_tpu.sh"
  rc=$?
  echo "=== rs_tpu.sh attempt ${try} exit ${rc} $(date -Is) ==="
  case "$rc" in
    12|13) echo "  (preempted / lost host -- retrying, generations resume from disk)" ;;
    *) break ;;
  esac
  sleep 60
done

# Auditable final state: no muse QR and no muse TPU VM anywhere.
for z in $ALL_ZONES; do
  echo "--- muse QRs in ${z} ---"
  timeout 120 "$GC" compute tpus queued-resources list --zone "$z" \
    --format="value(name,state.state)" 2>/dev/null | grep -i muse || echo "(none)"
  echo "--- muse TPU VMs in ${z} ---"
  timeout 120 "$GC" compute tpus tpu-vm list --zone "$z" \
    --format="value(name,state)" 2>/dev/null | grep -i muse || echo "(none)"
done
echo "=== cumulative slice seconds: $(cat "${REPO}/runs/muse_glimmer/rs2.slice_seconds" 2>/dev/null || echo 0) ==="
exit $rc
