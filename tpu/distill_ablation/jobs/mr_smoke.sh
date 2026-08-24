#!/bin/bash
# Exp-1 SMOKE: both map-reduce arms at N=3 on fc46. Validates the submit()-gated rollout loop,
# the digest+reduce path, and (for native) that we can actually count spawned subagents.
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB/portfolio"
for ARM in mechanical native; do
  echo "########## ARM $ARM ##########"
  "$PY" -u mapreduce.py --problem fc46 --arm $ARM --n 3 --replicate 99 \
      --rollout-wall 900 --orch-wall 1500 --max-parallel 3 --grader-conc 8 --topk 2 \
      || echo "^^ $ARM FAILED"
done
