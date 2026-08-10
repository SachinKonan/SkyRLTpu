#!/bin/bash
# Build all four RQ1 problem packs (grades each seed once at production budget).
# Wall: fc46 ~3min, erdos/ac1/ud ~20min each (serial) -> 1.5h is safe.
#SBATCH --job-name=rq1_packs
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq1/logs/%x_%j.log
set -euo pipefail
PY=/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover/.venv-ttd-discover/bin/python
RQ1=/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/rq1
export TTD_EVAL_BACKEND=local TTD_DISCOVER_SYNC=0
mkdir -p /n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq1/logs
$PY "$RQ1/server/make_problem_pack.py" --problems "${@:-fc46 erdos ac1 ud}"
