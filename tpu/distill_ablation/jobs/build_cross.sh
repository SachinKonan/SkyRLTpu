#!/bin/bash
# A4 corpus: cross-origin betters from the ensemble 20b+20b pool (worse=alpha,
# betters=beta), 20b teacher. Target >=32 survivors. Paid teacher gen (~$0.3).
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
ENS=$REPO/runs/ttd_ens_20b20b/tinker_log/erdos-ens-20b20b/puct_sampler_step_000015.json
cd "$AB"
"$PY" -u build_corpus.py --mode cross --name cross --self-tag alpha \
  --snapshot "$ENS" --heldout "$AB/heldout_seeds.json" \
  --teacher-model openai/gpt-oss-20b \
  --max-pairs 90 --teacher-phase1-tokens 26000 --max-target-tokens 8192 --chunk 32 \
  --out "$AB/corpora/cross.json"
