#!/bin/bash
# Slurm wrapper for the Muse-Glimmer follow-up TPU run.
#
# `--signal=TERM@300` deliberately WITHOUT the `B:` prefix, so the TERM reaches
# followup_tpu.sh itself and its own trap runs (a `setsid nohup` QR delete that
# outlives the shell).  `B:TERM` would signal only this wrapper and the QR would
# leak -- the exact failure this repo has been bitten by twice.
#
#SBATCH --job-name=mg-follow
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --signal=TERM@300
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/followup-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud
export PATH=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin:$HOME/.local/bin:$PATH
export CLOUDSDK_CORE_DISABLE_PROMPTS=1

echo "=== node $(hostname) job ${SLURM_JOB_ID:-manual} $(date -Is) ==="

QR="${QR:-sk7524-museglimmer-followup}"
ZONE="${ZONE:-us-east5-a}"
backstop() {
  echo "=== wrapper backstop: ensuring ${QR} is gone ==="
  setsid nohup "$GC" alpha compute tpus queued-resources delete "$QR" \
    --project=vision-mix --zone="$ZONE" --quiet --force \
    >>"${REPO}/runs/muse_glimmer/qr-delete.log" 2>&1 </dev/null &
  disown 2>/dev/null || true
  sleep 25
}
trap 'backstop' EXIT TERM INT HUP

bash "${REPO}/tpu/muse_glimmer/followup_tpu.sh"
rc=$?
echo "=== followup_tpu.sh exit ${rc} $(date -Is) ==="

# Auditable final state, printed into the job log: no muse QR and no muse TPU
# VM anywhere in the three zones this project uses.
for z in us-east5-a us-east5-b us-east5-c; do
  echo "--- muse QRs in ${z} ---"
  timeout 120 "$GC" compute tpus queued-resources list --zone "$z" \
    --format="value(name,state.state)" 2>/dev/null | grep -i muse || echo "(none)"
  echo "--- muse TPU VMs in ${z} ---"
  timeout 120 "$GC" compute tpus tpu-vm list --zone "$z" \
    --format="value(name,state)" 2>/dev/null | grep -i muse || echo "(none)"
done
exit $rc
