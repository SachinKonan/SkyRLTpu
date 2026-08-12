#!/bin/bash
# Real-weight HF -> MaxText/orbax conversion for muse-glimmer-30b, then push to
# GCS so the TPU slice can pull it in ~90 s instead of converting on a 97 GB
# boot disk that is already 60 GB full.
#
#SBATCH --job-name=mg-conv
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=08:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/convert-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
MTVENV=/n/fs/vision-mix/sk7524/muse-parity/mtvenv
HF=/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b
OUT=/n/fs/vision-mix/sk7524/caches/muse-glimmer-orbax
GCS=gs://sk7524-tinker-tpu-us-east5/muse-glimmer/maxtext-orbax
GC=/n/fs/vision-mix/sk7524/google-cloud-sdk/bin/gcloud

export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export HF_HOME=/n/fs/vision-mix/sk7524/caches/.hf
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

echo "=== node $(hostname) job ${SLURM_JOB_ID:-manual} $(date -Is) ==="

if [ -d "${OUT}/0/items" ]; then
  echo "orbax checkpoint already present at ${OUT}/0/items -- skipping convert"
else
  "${MTVENV}/bin/python" "${REPO}/tpu/muse_glimmer/convert_real.py" \
    --hf-dir "$HF" --out "$OUT" --model-name muse-glimmer-30b 2>&1
  rc=$?
  echo "=== convert exit ${rc} ==="
  [ "$rc" = 0 ] || exit "$rc"
fi

du -sh "${OUT}/0/items"
echo "=== pushing to ${GCS} ==="
time "$GC" storage rsync -r "${OUT}/0/items" "${GCS}/0/items" 2>&1 | tail -5
"$GC" storage du -s "${GCS}/0/items" 2>&1 | tail -2
echo "=== done $(date -Is) ==="
