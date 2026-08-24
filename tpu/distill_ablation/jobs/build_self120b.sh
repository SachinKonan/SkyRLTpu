#!/bin/bash
# A5 corpus: self-origin from the SAME ctrl15 pool as A2, but critiques written by
# the 120b teacher (teacher-quality axis). Target >=32 survivors. Paid 120b gen.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
cd "$AB"
"$PY" -u build_corpus.py --mode self --name self120b \
  --snapshot "$CTRL" --heldout "$AB/heldout_seeds.json" \
  --teacher-model openai/gpt-oss-120b \
  --max-pairs 90 --teacher-phase1-tokens 26000 --max-target-tokens 8192 --chunk 32 \
  --out "$AB/corpora/self120b.json"
