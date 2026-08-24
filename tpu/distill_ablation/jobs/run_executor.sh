#!/bin/bash
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB"
"$PY" -u run_executor.py --plans "$AB/corpora/${AB_PLANS}" --executor-model "$AB_MODEL" \
  --tag "$AB_TAG" --exec-concurrency 4 --exec-timeout 300 \
  --worse-set "$AB/corpora/worse_set.json" --pool-snapshot "$CTRL" \
  --out "$AB/corpora/exec_${AB_TAG}.json"
