#!/bin/bash
# CPU control for the prompt-ladder run. No chips, no engines: prove the
# extraction -> compose -> pre-gate path and the P4 primitives before any QR.
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0:50:00
#SBATCH --job-name=ladder-control
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/ladder-control-%j.log
set -uo pipefail
repo=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$repo"
export PYTHONPATH="${repo}/tpu:${PYTHONPATH:-}"
mkdir -p runs/pallas_arena
JAX_PLATFORMS=cpu uv run --isolated --extra dev --extra jax --with flatbuffers \
  python -m pallas_arena.probe.ladder_control \
  --out "runs/pallas_arena/ladder-control-${SLURM_JOB_ID}.json"
echo "control rc=$?"
