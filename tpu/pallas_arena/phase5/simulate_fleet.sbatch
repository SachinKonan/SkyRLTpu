#!/bin/bash
# Phase-5 fleet simulation on CPU: zero TPU, everything but the silicon.
# Must be green before any QR is created.
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH -t 1:00:00
#SBATCH --job-name=pallas-arena-phase5-sim
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/phase5-sim-%j.log

set -uo pipefail
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run under SLURM (sbatch)." >&2
  exit 1
fi

repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
out_dir="${repo_root}/runs/pallas_arena"
mkdir -p "$out_dir"
cd "$repo_root"
export PYTHONPATH="${repo_root}/tpu:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu

echo "=== phase-5 fleet simulation ($(date +%H:%M:%S), job ${SLURM_JOB_ID}) ==="
uv run --isolated --extra dev --extra jax --with fastapi --with uvicorn --with flatbuffers \
  python tpu/pallas_arena/phase5/simulate_fleet.py \
  --judges "${JUDGES:-5}" --target "${TARGET:-400}" --dups "${DUPS:-40}" \
  --mock-grade-s "${MOCK_GRADE_S:-0.5}" \
  --out "${out_dir}/phase5-sim-results-${SLURM_JOB_ID}.json"
rc=$?
echo "=== simulation done rc=${rc} ($(date +%H:%M:%S)) ==="
exit "$rc"
