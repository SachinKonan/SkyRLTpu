#!/bin/bash
# Small cross-model corpus sample: gpt-oss worse <- Qwen better, gpt-oss teacher.
# Prints the surviving critiques so we can judge teacher comprehension. ~$0.3.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
cd "$AB"
"$PY" -u build_corpus.py \
  --foreign-betters "$AB/corpora/foreign_betters.json" --better-origin qwen \
  --snapshot "$CTRL" --name foreign_qwen_sample --heldout "$AB/heldout_seeds.json" \
  --max-pairs 40 --teacher-phase1-tokens 26000 --max-target-tokens 8192 --chunk 20 \
  --show-critiques 8 \
  --out "$AB/corpora/foreign_qwen_sample.json"
