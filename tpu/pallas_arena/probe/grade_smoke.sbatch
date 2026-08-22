#!/bin/bash
# Grade one evolve-smoke generations JSONL on a CPU node (export-child
# validity + interpret-mode correctness -- never on the login node).
#
# Submit:  GENS_JOB=<jobid> sbatch tpu/pallas_arena/probe/grade_smoke.sbatch
# Logs:    runs/pallas_arena/grade-smoke-<jobid>.log
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o /n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/grade-smoke-%j.log
set -euo pipefail
cd /n/fs/vision-mix/sk7524/SkyRLTpu/tpu
export ARENA_RLIMIT_GB=64
uv run --isolated --extra dev --extra jax --extra tinker --with flatbuffers python -m pallas_arena.probe.grade_smoke \
  --gens "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/evolve-smoke-gens-${GENS_JOB}.jsonl" \
  --out "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/evolve-smoke-graded-${GENS_JOB}.json"
