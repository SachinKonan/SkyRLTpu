#!/bin/bash
# Slurm wrapper for the Muse-Glimmer TP head-to-head (one TP=4 engine vs two
# TP=2 engines on the same 4 chips).
#
# `--signal=TERM@300` deliberately WITHOUT the `B:` prefix so the TERM reaches
# tp_compare_tpu.sh itself and its own trap runs; `B:TERM` would signal only
# this wrapper and the QR would leak.
#
#SBATCH --job-name=mg-tp
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --signal=TERM@300
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/tpcompare-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
export PATH=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin:$HOME/.local/bin:$PATH
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

echo "=== node $(hostname) job ${SLURM_JOB_ID:-manual} $(date -Is) ==="

QR="${QR:-sk7524-museglimmer-tp}"
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

bash "${REPO}/tpu/muse_glimmer/tp_compare_tpu.sh"
rc=$?
echo "=== tp_compare_tpu.sh exit ${rc} $(date -Is) ==="

# Auditable final state: no muse QR and no muse TPU VM anywhere.
for z in $ALL_ZONES; do
  echo "--- muse QRs in ${z} ---"
  timeout 120 "$GC" compute tpus queued-resources list --zone "$z" \
    --format="value(name,state.state)" 2>/dev/null | grep -i muse || echo "(none)"
  echo "--- muse TPU VMs in ${z} ---"
  timeout 120 "$GC" compute tpus tpu-vm list --zone "$z" \
    --format="value(name,state)" 2>/dev/null | grep -i muse || echo "(none)"
done
exit $rc
