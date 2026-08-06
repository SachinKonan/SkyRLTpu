#!/bin/bash
# Grade one RQ1 run dir:  sbatch grade.sh <problem> <run_dir> [concurrency]
# fc46@200 ~25min; python problems worst-case ~2.5-3h at 24-way.
#SBATCH --job-name=rq1_grade
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=05:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq1/logs/%x_%j.log
set -euo pipefail
PY=/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover/.venv-ttd-discover/bin/python
RQ1=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/tpu/rq1
export TTD_EVAL_BACKEND=local TTD_DISCOVER_SYNC=0
mkdir -p /n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq1/logs
$PY "$RQ1/server/grade_batch.py" --problem "$1" --run-dir "$2" --concurrency "${3:-24}"
