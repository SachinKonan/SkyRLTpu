#!/bin/bash
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB"
echo "[bench] args: ${AB_BENCH_ARGS:-}"
exec "$PY" -u benchmark_cdc.py ${AB_BENCH_ARGS:-}
