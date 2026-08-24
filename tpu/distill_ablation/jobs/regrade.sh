#!/bin/bash
# Re-grade all executor outputs from the cached workdirs (authoritative; the
# in-process got_code/c5 undercounts because run_executor discarded solution.py on
# codex TIMEOUT). Grades locally on a compute node across all cores.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
cd "$AB"
for pair in "van_mini:gpt-5.4-mini" "van_terra:gpt-5.6-terra" "van_luna:gpt-5.6-luna"; do
  tag="${pair%%:*}"; model="${pair##*:}"
  echo "########## REGRADE $tag ($model) ##########"
  "$PY" -u regrade_from_disk.py --tag "$tag" --executor "$model" \
    --worse-set "$AB/corpora/worse_set.json" --eval-timeout 1100 --concurrency 14 \
    --out "$AB/corpora/regrade_${tag}.json"
done
