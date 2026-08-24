#!/bin/bash
# Generate + store teacher critiques for one FAIR arm (shared worse set x source
# betters). AB_SOURCE = own | qwen | nemo. Paid once per arm.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
SRC="${AB_SOURCE:?set AB_SOURCE}"
cd "$AB"
EXTRA=""
[ "$SRC" != "own" ] && EXTRA="--foreign-betters $AB/corpora/foreign_betters.json"
"$PY" -u gen_teacher_outputs.py --name "teacher_fair_${SRC}" \
  --snapshot "$CTRL" --heldout "$AB/heldout_seeds.json" \
  --worse-set "$AB/corpora/worse_set.json" --source "$SRC" $EXTRA \
  --teacher-phase1-tokens 26000 --chunk 20 \
  --out "$AB/corpora/teacher_fair_${SRC}.json"
