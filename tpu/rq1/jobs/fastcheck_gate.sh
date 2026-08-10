#!/bin/bash
#SBATCH --job-name=rq2_gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=03:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq1/logs/%x_%j.log
set -euo pipefail
PY=/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover/.venv-ttd-discover/bin/python
export TTD_EVAL_BACKEND=local TTD_DISCOVER_SYNC=0
$PY /n/fs/vision-mix/sk7524/SkyRLTpu/tpu/rq1/server/fastcheck_gate.py \
  --problem "$1" --run-dir "$2" --concurrency "${3:-24}"
