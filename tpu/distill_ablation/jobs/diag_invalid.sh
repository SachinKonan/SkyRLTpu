#!/bin/bash
# Diagnose why invalid gens failed. AB_INITIAL, AB_ENV, AB_PT, AB_POOL (opt), AB_BUDGET.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"; cd "$AB"
exec "$PY" -u diagnose_invalid.py --initial "${AB_INITIAL:?}" \
  --env "${AB_ENV:?}" --problem-type "${AB_PT:-}" --budget "${AB_BUDGET:-20}" --eval-timeout 90 \
  ${AB_POOL:+--pool-snapshot "$AB_POOL"}
