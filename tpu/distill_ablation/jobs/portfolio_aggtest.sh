#!/bin/bash
# One resumed round to exercise the AGGREGATOR path (store has >=3 nodes).
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB/portfolio"
exec "$PY" -u loop.py --problem erdos --tag _smoke --rounds 1 --n-groups 2 --g-rollouts 1 --exec-wall 900 --agg-wall 600
