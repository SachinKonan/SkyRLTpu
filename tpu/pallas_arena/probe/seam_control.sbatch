#!/bin/bash
# CPU control battery for the seam. No chips, no QRs, no network.
#   1. every seam's known-good fill through extract -> compose -> CPU AOT
#      pre-gate at every declared PROBE shape (including the non-divisible
#      holdout), exporting FOR the tpu platform from a CPU host;
#   2. CPU numerics (`interpret=True` for the pallas ones) against each
#      problem's own fp32 reference, plus the FLCE gradient;
#   3. the seam dry run: prompts, scaffold identity, extractor, metrics.
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 1:30:00
#SBATCH --job-name=seam-control
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/seam-control-%j.log

set -uo pipefail
repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
out_dir="${repo_root}/runs/pallas_arena"
mkdir -p "$out_dir"
cd "$repo_root"
export PYTHONPATH="${repo_root}/tpu:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu

TASKS=${TASKS:-}
rc=0

echo "=== seam dry run $(date +%H:%M:%S) ==="
uv run --isolated --extra dev --extra jax --with flatbuffers \
  python tpu/pallas_arena/probe/seam_dryrun.py 2>&1 | tail -120
rc=$(( rc + ${PIPESTATUS[0]} ))

echo
echo "=== seam control: compose -> AOT export -> numerics $(date +%H:%M:%S) ==="
uv run --isolated --extra dev --extra jax --with flatbuffers \
  python tpu/pallas_arena/probe/seam_control.py --interpret \
  ${TASKS:+--tasks "$TASKS"} \
  --out "${out_dir}/seam-control-${SLURM_JOB_ID}.json" 2>&1 | tail -200
rc=$(( rc + ${PIPESTATUS[0]} ))

echo
echo "=== arena regression battery $(date +%H:%M:%S) ==="
uv run --isolated --extra dev --extra jax --with fastapi --with uvicorn --with httpx --with flatbuffers \
  python -m pytest tpu/pallas_arena/tests -q 2>&1 | tail -40
rc=$(( rc + ${PIPESTATUS[0]} ))

echo "=== seam control done rc=${rc} $(date +%H:%M:%S) ==="
exit "$rc"
