#!/bin/bash
# Initial-behavior probe for a frontier_algo problem: gpt-oss-120b, C++17, base prompt.
# AB_PT (problem id: 46/302), AB_TAG, AB_NGENS.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
FCROOT=/n/fs/vision-mix/sk7524/SkyRLTpu-frontiercs/third_party/discover
PY="${TTD_AB_VENV}/bin/python"
export TTD_FCALGO_MAX_CASES=0           # all cases (46=1, 302=10)
export PYTHONPATH="$FCROOT:${PYTHONPATH:-}"
cd "$AB"
exec "$PY" -u gen_initial.py --env frontier_algo --problem-type "${AB_PT:?}" \
  --tag "${AB_TAG:?}" --model openai/gpt-oss-120b --n-gens "${AB_NGENS:-24}" \
  --phase1-max-tokens 26000 --eval-timeout 150 --grade-concurrency 12 \
  --discover-root "$FCROOT" --code-lang cpp \
  --out "$AB/corpora/initial_${AB_TAG}.json"
