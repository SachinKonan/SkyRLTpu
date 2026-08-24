#!/bin/bash
# PILOT: fc46, three aggregation arms, matched compute.
#   10 states x 10 rollouts = 100/round, 10 rounds = 1000 mini rollouts per arm.
# Everything except the aggregation step is identical across arms.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB/portfolio"

COMMON="--problem fc46 --states 10 --rollouts 10 --rounds 10 \
        --max-parallel 25 --grader-conc 32 \
        --rollout-wall 900 --coord-wall 1200 --max-sessions 1100"

echo "########## ARM 1/3: simple_tes + rpucg formula ##########"
"$PY" -u sweep.py $COMMON --reuse simple_tes --select rpucg --tag _rpucg || echo "^^ ARM FAILED"

echo "########## ARM 2/3: continual (shared workspace) ##########"
"$PY" -u sweep.py $COMMON --reuse continual --tag _cont || echo "^^ ARM FAILED"

echo "########## ARM 3/3: simple_tes + agent selection ##########"
"$PY" -u sweep.py $COMMON --reuse simple_tes --select agent --tag _agent || echo "^^ ARM FAILED"

echo "########## ALL ARMS DONE ##########"
for T in _rpucg _cont _agent; do
  echo "--- fc46$T"; cat "$REPO/runs/sweep/fc46_"*"$T/trajectory.json" 2>/dev/null | head -40
done
