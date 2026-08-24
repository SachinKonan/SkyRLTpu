#!/bin/bash
# Validity check: base gpt-oss-20b, eval-only, 3 seeds x 2 rollouts at the REAL
# 26000 budget. Confirms base produces valid gradeable improver code before the
# full Phase 1 spend. ~$0.1.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
cd "$AB"
"$PY" -u finetune_and_eval.py --arm A0VALID --eval-only \
  --k 2 --max-seeds 3 --phase1-max-tokens 26000 \
  --heldout "$AB/heldout_seeds.json" --eval-pool-snapshot "$CTRL" \
  --out-dir "$REPO/runs/distill_ablation/A0VALID"
