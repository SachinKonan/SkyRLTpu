#!/bin/bash
# One fair arm: finetune base gpt-oss on fair_<src> (count-matched) + best-of-K eval.
# AB_SOURCE = own | qwen | nemo.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
SRC="${AB_SOURCE:?set AB_SOURCE}"
cd "$AB"
exec "$PY" -u finetune_and_eval.py --arm "fair_${SRC}" \
  --corpus "$AB/corpora/fair_${SRC}.json" --subset-size 27 --epochs 8 \
  --probe "$AB/corpora/fair_${SRC}.json" \
  --k 8 --phase1-max-tokens 26000 --eval-concurrency 8 \
  --heldout "$AB/heldout_seeds.json" --eval-pool-snapshot "$CTRL" \
  --out-dir "$REPO/runs/distill_ablation/fair_${SRC}"
