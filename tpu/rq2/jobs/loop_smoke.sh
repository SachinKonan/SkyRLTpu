#!/bin/bash
# Cheapest possible end-to-end exercise of the loop: fc46 (grades in ~3s/program),
# tiny n, 2 steps. Validates render -> sample -> two-phase -> grade -> reflect -> aggregate.
#   sbatch loop_smoke.sh <state> <execution> <composition> [n] [steps]
#SBATCH --job-name=rq2_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --exclude=neu301
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2/logs/%x_%j.log
set -uo pipefail
RQ2=/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/rq2
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
st="${1:-puct}"; ex="${2:-simple}"; comp="${3:-qwen}"; n="${4:-8}"; steps="${5:-2}"
# NOT `uv run`: loop.py grades inline (ttt_discover, numpy) and its PUCT state uses
# store.py (networkx), none of which live in a PEP 723 minimal env. The discover venv has
# httpx, networkx, numpy and mcp already.
PY=/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover/.venv-ttd-discover/bin/python
export TTD_EVAL_BACKEND=local TTD_DISCOVER_SYNC=0
cd "$RQ2/client"
$PY loop.py --problem fc46 --state "$st" --execution "$ex" --composition "$comp" \
  --n "$n" --steps "$steps" --concurrency 8 --grade-concurrency 8 \
  --out "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2/smoke/fc46_${st}_${ex}_${comp}"
