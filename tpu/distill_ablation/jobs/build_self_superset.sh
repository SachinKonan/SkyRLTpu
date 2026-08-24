#!/bin/bash
# Build the self-origin superset corpus (Phase 1 dose sweep subsamples it).
# Over-generate to ~128+ survivors. Paid teacher gen (20b). ~$1.5, ~1h.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
cd "$AB"
"$PY" -u build_corpus.py --mode self --name self_superset \
  --snapshot "$CTRL" --heldout "$AB/heldout_seeds.json" \
  --max-pairs 400 --teacher-phase1-tokens 26000 --max-target-tokens 8192 --chunk 32 \
  --out "$AB/corpora/self_superset.json"
