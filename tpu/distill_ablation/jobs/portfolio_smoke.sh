#!/bin/bash
# Portfolio-search SMOKE: erdos, 2 rounds x 2 groups x 1 rollout (~4 mini + 1 sol session).
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB/portfolio"
exec "$PY" -u loop.py --problem erdos --tag _smoke \
    --rounds 2 --n-groups 2 --g-rollouts 1 \
    --exec-wall 900 --agg-wall 600
