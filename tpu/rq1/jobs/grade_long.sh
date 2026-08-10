#!/bin/bash
# Long-wall grading for the heavy problems (ud, erdos): every candidate burns its full 1000s
# production budget, so 200 programs is many CPU-hours and the 5h wall of grade.sh is not
# enough (ud_D hit TIMEOUT). grade_batch caches per-program results, so a rerun resumes.
#   sbatch grade_long.sh <problem> <run_dir> [concurrency]
#SBATCH --job-name=rq1_gradeL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=23:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq1/logs/%x_%j.log
set -euo pipefail
PY=/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover/.venv-ttd-discover/bin/python
RQ1=/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/rq1
export TTD_EVAL_BACKEND=local TTD_DISCOVER_SYNC=0
$PY "$RQ1/server/grade_batch.py" --problem "$1" --run-dir "$2" --concurrency "${3:-30}"
