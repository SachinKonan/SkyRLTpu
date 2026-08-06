#!/bin/bash
# Neuronic-side T1 smoke/run (Meta devserver is the real home for cell B; this wrapper exists
# for shakedown + fallback). Pin an egress-verified node: neu301 drops websockets, neu332 clean.
#   sbatch --nodelist=neu332 collect_t1_neuronic.sh <problem> <out> [n] [concurrency]
#SBATCH --job-name=rq1_t1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --exclude=neu301
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq1/logs/%x_%j.log
set -euo pipefail
RQ1=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/tpu/rq1
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
mkdir -p /n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq1/logs
cd "$RQ1/client"
uv run collect_t1.py --problem "$1" --out "$2" --n "${3:-3}" \
  --concurrency "${4:-3}" --site neuronic --resume
