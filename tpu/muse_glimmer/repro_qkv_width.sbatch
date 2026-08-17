#!/bin/bash
#SBATCH --job-name=mg-reproqkv
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/reproqkv-%j.log
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
set -uo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
MGVENV=/n/fs/vision-mix/sk7524/muse-vllm/venv

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-none}  $(date -Is) ==="
export PYTHONPATH="${REPO}/third_party/tpu-inference:${PYTHONPATH:-}"
"${MGVENV}/bin/python" "${REPO}/tpu/muse_glimmer/repro_qkv_width.py"
rc=$?
echo "=== exit ${rc}  $(date -Is) ==="
exit ${rc}
