#!/bin/bash
# SMOKE: fc46, tiny scale, all three aggregation paths.
#   continual          -> coordinator distills the round digest into a shared workspace
#   simple_tes/agent   -> coordinator picks seed groups via store MCP (constraints enforced)
#   simple_tes/rpucg   -> formula selector, no LLM (free; validates the DAG math end-to-end)
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB/portfolio"
run () {
  echo "########## $1 ##########"
  shift
  "$PY" -u sweep.py --problem fc46 --tag "${TAG:-_smoke}" \
      --states 2 --rollouts 2 --rounds 2 \
      --max-parallel 4 --rollout-wall 600 --coord-wall 600 \
      --grader-conc 8 --max-sessions 40 "$@" || echo "^^ FAILED"
}
TAG=_smoke_rpucg run "simple_tes / rpucg (no LLM selector)" --reuse simple_tes --select rpucg
run "continual"                            --reuse continual
run "simple_tes / agent"                   --reuse simple_tes --select agent
