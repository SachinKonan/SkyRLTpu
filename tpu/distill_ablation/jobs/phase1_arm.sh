#!/bin/bash
# One Phase-1 arm. Parameterized by env: AB_ARM, AB_SUBSET, AB_EVAL_ONLY.
#   A0: AB_EVAL_ONLY=1                 (base, no finetune)
#   A1/A2/A3: AB_SUBSET=8/32/128       (CE finetune then eval)
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
HO=$AB/heldout_seeds.json
CORPUS=$AB/corpora/self_superset.json
cd "$AB"

COMMON="--k 8 --phase1-max-tokens 26000 --eval-concurrency 8 \
  --heldout $HO --eval-pool-snapshot $CTRL --probe $CORPUS \
  --out-dir $REPO/runs/distill_ablation/${AB_ARM}"

if [ "${AB_EVAL_ONLY:-0}" = "1" ]; then
  exec "$PY" -u finetune_and_eval.py --arm "$AB_ARM" --eval-only $COMMON
else
  exec "$PY" -u finetune_and_eval.py --arm "$AB_ARM" \
    --corpus "$CORPUS" --subset-size "$AB_SUBSET" --epochs 8 $COMMON
fi
