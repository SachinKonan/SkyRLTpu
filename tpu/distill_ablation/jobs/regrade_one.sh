#!/bin/bash
# Re-grade ONE executor tag from its cached workdirs, at high concurrency (1 wave).
# AB_TAG (van_mini|van_terra|van_luna), AB_MODEL (label), AB_CONC (concurrency).
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
cd "$AB"
exec "$PY" -u regrade_from_disk.py --tag "${AB_TAG:?}" --executor "${AB_MODEL:?}" \
  --worse-set "$AB/corpora/worse_set.json" --eval-timeout 1100 --concurrency "${AB_CONC:?}" \
  --out "$AB/corpora/regrade_${AB_TAG}.json"
