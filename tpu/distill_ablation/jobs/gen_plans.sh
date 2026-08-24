#!/bin/bash
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
cd "$AB"
"$PY" -u gen_plans.py --context "${AB_CTX}" --n-bases 16 --n-plans 3 \
  --worse-set "$AB/corpora/worse_set.json" --foreign-betters "$AB/corpora/foreign_betters.json" \
  --pool-snapshot "$CTRL" --out "$AB/corpora/plans_${AB_CTX}.json"
