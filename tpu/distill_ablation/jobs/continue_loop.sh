#!/bin/bash
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB"
echo "[continue] args: ${AB_LOOP_ARGS:-}"
exec "$PY" -u continue_loop.py ${AB_LOOP_ARGS:-}
