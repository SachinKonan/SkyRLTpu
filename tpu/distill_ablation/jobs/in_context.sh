#!/bin/bash
# One in-context-improvement arm. AB_ARM = foreign | own | vanilla.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
SRC="${AB_ARM:?set AB_ARM}"
cd "$AB"
exec "$PY" -u in_context_improve.py --arm "$SRC" \
  --n-bases 16 --n-gens 5 --phase1-max-tokens 26000 --eval-timeout 1100 \
  --base-concurrency 4 \
  --worse-set "$AB/corpora/worse_set.json" \
  --foreign-betters "$AB/corpora/foreign_betters.json" \
  --pool-snapshot "$CTRL" \
  --out "$AB/corpora/incontext_${SRC}.json"
