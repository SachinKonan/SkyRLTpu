#!/bin/bash
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
cd "$AB"
"$PY" -u in_context_improve.py --arm "${AB_ARM}" --n-bases 16 --n-gens 3 \
  --phase1-max-tokens 26000 --eval-timeout 1100 --base-concurrency 4 \
  --worse-set "$AB/corpora/worse_set.json" --pool-snapshot "$CTRL" \
  --out "$AB/corpora/direct_${AB_ARM}.json"
