#!/bin/bash
# Stage the real Muse-Glimmer-30B weights: HF hub -> local NFS -> GCS mirror.
#
# The 30B index has only TWO shards and BOTH are mixed text+vision (627 text /
# 809 vision tensors), so a "text shards only" download is not possible -- the
# whole 59.5 GB snapshot comes down.
#
# Downloaded with --local-dir (a plain snapshot directory, ONE copy on disk).
# huggingface_hub 1.27 no longer uses the blobs/+symlinked-snapshots layout
# (it writes refs/ + trees/ + an xet store), so mirroring "a hub cache dir"
# verbatim would no longer be portable. Instead the GCS side is written in the
# CLASSIC hub layout -- refs/main + snapshots/<sha>/<files> with real files --
# which is what `gsutil -m rsync -r $HF_CACHE_GCS $HF_HOME/hub` in
# tpu/start_vllm_tpu.sh reconstitutes, and which any HF version can read.
#
#SBATCH --job-name=mg-stage
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/stage-%j.log

set -uo pipefail
REPO=meta-models/Muse-Glimmer-30B
SHA=a4e59da52a7bc87ae7251dd5545c0dd437c44b68
LOCAL=/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b
GCS=gs://sk7524-tinker-tpu-us-east5/hf-cache/models--meta-models--Muse-Glimmer-30B
VENV=/n/fs/vision-mix/sk7524/caches/muse-parity/venv

export HF_HOME=/n/fs/vision-mix/sk7524/caches/.hf
export HF_HUB_OFFLINE=0
mkdir -p /n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer "${LOCAL}"

echo "=== node $(hostname) job ${SLURM_JOB_ID:-manual} $(date -Is) ==="
df -h /n/fs/vision-mix | tail -1

echo "=== [1/3] download ${REPO} -> ${LOCAL} ==="
time "${VENV}/bin/hf" download "${REPO}" --revision "${SHA}" \
  --local-dir "${LOCAL}" --max-workers 16 \
  || { echo "FATAL: download failed"; exit 3; }

echo "=== download done $(date -Is) ==="
du -sh "${LOCAL}"
ls -la "${LOCAL}" | head -30

echo "=== [2/3] mirror -> ${GCS} (classic hub layout) ==="
printf '%s' "${SHA}" > "${LOCAL}/.refs_main"
gsutil -q cp "${LOCAL}/.refs_main" "${GCS}/refs/main" || { echo "FATAL: refs upload"; exit 4; }
gsutil -o 'GSUtil:parallel_composite_upload_threshold=150M' \
       -o 'GSUtil:parallel_process_count=8' -o 'GSUtil:parallel_thread_count=8' \
       -m rsync -r -x '^\.cache/|^\.refs_main$' "${LOCAL}" "${GCS}/snapshots/${SHA}" \
  || { echo "FATAL: gcs mirror failed"; exit 5; }

echo "=== [3/3] verify ==="
gsutil du -s "${GCS}"
gsutil ls "${GCS}/snapshots/${SHA}/"
echo "=== STAGE DONE $(date -Is) ==="
