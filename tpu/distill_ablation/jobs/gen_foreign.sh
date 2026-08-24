#!/bin/bash
# Generate the foreign-better pool: Nemotron + Qwen high-scoring Erdos programs,
# saved as origin-tagged States for cross-model distillation.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
cd "$AB"
"$PY" -u gen_foreign_betters.py --models nemo qwen \
  --n-seeds 5 --n-samples 8 --max-tokens 26000 --eval-timeout 1100 \
  --keep-c5-below 0.3835 --pool-snapshot "$CTRL" \
  --heldout "$AB/heldout_seeds.json" \
  --out "$AB/corpora/foreign_betters.json"
