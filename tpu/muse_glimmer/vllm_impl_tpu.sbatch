#!/bin/bash
# Driver for the torch/vLLM Muse-Glimmer slice run.
#
# Runs on a CPU node -- all it does is drive gcloud and ssh; the TPU work
# happens on the slice.  The slurm time limit is deliberately LONGER than the
# script's own CAP_SEC so that the script's trapped cleanup, not a slurm kill,
# is what ends the run: a scancel has previously killed a cleanup mid-delete
# and leaked a QR.
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -t 6:00:00
#SBATCH --job-name=mg-vllmimpl
#SBATCH --signal=B:TERM@600
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/vllmimpl-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
export PATH=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin:$PATH
# gcloud file logging was disabled deliberately (home has <2G free); leave off.
export CLOUDSDK_CORE_DISABLE_FILE_LOGGING=1
export CLOUDSDK_PYTHON_SITEPACKAGES=1

echo "=== driver node $(hostname)  job ${SLURM_JOB_ID:-manual}  $(date -Is) ==="

# Forward slurm's pre-timeout TERM to the script so its own trap runs the QR
# teardown while there is still time.
child=""
forward() { echo "driver: forwarding TERM to ${child}"; [ -n "$child" ] && kill -TERM "$child" 2>/dev/null; wait "$child"; exit 143; }
trap forward TERM INT

bash "${REPO}/tpu/muse_glimmer/vllm_impl_tpu.sh" "$@" &
child=$!
wait "$child"
rc=$?
echo "=== vllm_impl_tpu.sh exit ${rc}  $(date -Is) ==="
exit $rc
