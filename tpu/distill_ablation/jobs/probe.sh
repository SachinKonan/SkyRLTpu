#!/bin/bash
# Foreign-model probe: Qwen3.6 + Nemotron-120B write Erdos programs -> grade. ~$0.3
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
cd "$AB"
"$PY" -u probe_foreign.py \
  --n-seeds 3 --n-samples 6 --max-tokens 26000 --eval-timeout 1100 \
  --heldout "$AB/heldout_seeds.json" --eval-pool-snapshot "$CTRL"
