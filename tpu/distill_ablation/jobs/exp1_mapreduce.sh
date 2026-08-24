#!/bin/bash
# EXP-1: which map-reduce protocol is better, at N=50?
#   native      = codex multi_agent_v2 decides its own map/verify/reduce (permitted 50)
#   mechanical  = driver forks exactly 50 terra rollouts -> sol reduces a digest of their submissions
# Workers gpt-5.6-terra xhigh, orchestrator/reducer gpt-5.6-sol xhigh, every rollout submit()-gated
# so all scores are server-verified. C++ problems first (grade_full ~150s vs 1100s for python).
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB/portfolio"
for PROB in fc46 fc302; do
  for ARM in mechanical native; do
    echo "########## $PROB / $ARM ##########"
    "$PY" -u mapreduce.py --problem $PROB --arm $ARM --n 50 --replicate 0 \
        --rollout-wall 1200 --orch-wall 3600 --max-parallel 25 \
        --grader-conc 32 --max-full 3 --topk 5 || echo "^^ $PROB/$ARM FAILED"
  done
done
echo "########## EXP-1 DONE ##########"
for d in "$REPO"/runs/mapreduce/*_r0; do
  [ -f "$d/result.json" ] && cat "$d/result.json" | tr -d '\n' && echo
done
