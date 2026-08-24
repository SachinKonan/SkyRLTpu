#!/bin/bash
# Shared meta-round 1: expand the bare seed once per model, once per problem.
# Round 1 is identical for all six treatments (only the root exists, and it is the seed, so
# neither the selection policy nor the continuation rule can differ yet), so running it here and
# sharing it saves 5 x 5,120 rollouts per problem.
#   sbatch --array=0-1 tpu/rq2/jobs/meta_bootstrap.sbatch
#SBATCH --job-name=rq2_meta_boot
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --exclude=neu301
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2/logs/%x_%A_%a.log
set -uo pipefail
cd /n/fs/vision-mix/sk7524/SkyRLTpu
PROBLEMS=(erdos fc46)
P=${PROBLEMS[$SLURM_ARRAY_TASK_ID]}
export TTD_EVAL_BACKEND=local TTD_DISCOVER_SYNC=0
echo "[boot] problem=$P start=$(date '+%F %T')"
third_party/discover/.venv-ttd-discover/bin/python tpu/rq2/meta/driver.py \
  --problem "$P" --bootstrap --compact-model "${COMPACT_MODEL:-gpt-5.6-terra}"
echo "[boot] problem=$P done=$(date '+%F %T')"
