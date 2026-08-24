#!/bin/bash
# Initial-behavior probe: gpt-oss-120b, ttt-discover BASE prompt, one problem.
# AB_ENV, AB_PT (problem_type), AB_TAG, AB_POOL (optional pool snapshot for erdos).
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
cd "$AB"
exec "$PY" -u gen_initial.py --env "${AB_ENV:?}" --problem-type "${AB_PT:-}" \
  --tag "${AB_TAG:?}" --model openai/gpt-oss-120b --n-gens "${AB_NGENS:-24}" \
  --phase1-max-tokens 26000 --eval-timeout 1100 --grade-concurrency "${AB_GC:-12}" \
  ${AB_POOL:+--pool-snapshot "$AB_POOL"} \
  --out "$AB/corpora/initial_${AB_TAG}.json"
