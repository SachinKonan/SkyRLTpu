#!/bin/bash
# Generate + STORE all teacher critiques for gpt-oss<-Qwen cross pairs (paid once).
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
cd "$AB"
"$PY" -u gen_teacher_outputs.py --name teacher_qwen \
  --snapshot "$CTRL" --heldout "$AB/heldout_seeds.json" \
  --foreign-betters "$AB/corpora/foreign_betters.json" --better-origin qwen \
  --max-pairs 80 --teacher-phase1-tokens 26000 --chunk 20 \
  --out "$AB/corpora/teacher_qwen.json"
