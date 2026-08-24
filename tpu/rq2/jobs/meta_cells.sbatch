#!/bin/bash
# The 12 meta-search treatment cells: {a,b,c} continuation x {greedy,mcts} selection,
# on erdos and fc46, depth 5, two models.
#
# Each task runs ONE treatment to depth D. Within a meta-round the driver runs the two
# expansions (one per model) as concurrent subprocesses -- they hit disjoint worker pools
# (qwen vs gemma), so they do not contend with each other.
#
#   sbatch --dependency=afterok:<boot_jobid> --array=0-11%12 tpu/rq2/jobs/meta_cells.sbatch
#
# index = problem*6 + rule*2 + policy
#SBATCH --job-name=rq2_meta
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --exclude=neu301
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2/logs/%x_%A_%a.log
set -uo pipefail
cd /n/fs/vision-mix/sk7524/SkyRLTpu

PROBLEMS=(erdos fc46)
RULES=(a b c)
POLICIES=(greedy mcts)

i=$SLURM_ARRAY_TASK_ID
P=${PROBLEMS[$((i / 6))]}
R=${RULES[$(((i % 6) / 2))]}
POL=${POLICIES[$((i % 2))]}
DEPTH=${DEPTH:-5}

export TTD_EVAL_BACKEND=local TTD_DISCOVER_SYNC=0
echo "[meta] cell $i: problem=$P rule=$R policy=$POL depth=$DEPTH start=$(date '+%F %T')"
third_party/discover/.venv-ttd-discover/bin/python tpu/rq2/meta/driver.py \
  --problem "$P" --rule "$R" --policy "$POL" --depth "$DEPTH" \
  --compact-model "${COMPACT_MODEL:-gpt-5.6-terra}"
echo "[meta] cell $i done=$(date '+%F %T')"
