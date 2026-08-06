#!/bin/bash
# Phase 1 CPU test battery for the Pallas kernel arena judge.
# COMPUTE NODES ONLY — ~30 forked jax-CPU grading children (never login).
#
# Submit:  sbatch tpu/pallas_arena/run_phase1_tests.sbatch
# Logs:    runs/pallas_arena/phase1-tests-<jobid>.log
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH --job-name=pallas-arena-phase1
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/phase1-tests-%j.log

set -uo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run under SLURM (sbatch); no compute on the login node." >&2
  exit 1
fi

repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$repo_root"

export JAX_PLATFORMS=cpu

echo "=== pallas arena phase-1 battery ($(date +%H:%M:%S), job ${SLURM_JOB_ID}) ==="
uv run --isolated --extra dev --extra jax \
    --with fastapi --with httpx --with uvicorn --with flatbuffers \
    pytest tpu/pallas_arena/tests -v -rA --durations=15 \
    -p no:cacheprovider
rc=$?
echo "=== battery done rc=${rc} ($(date +%H:%M:%S)) ==="
exit "$rc"
