#!/bin/bash
# Scheme A/B: fast-check-only (10min) vs full-grader (30min), 10 parallel mini rollouts each,
# same weak 20b seed (worse_set[2]), identical 1100s production yardstick.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB/portfolio"
exec "$PY" -u ab_scheme.py --problem erdos --seed-idx 2 --arms fast full \
    --group 10 --model gpt-5.4-mini --reasoning high --max-concurrent 12
