#!/bin/bash
# EXP-1 (corrected): at an equal budget of 100 subagent calls / 25 concurrent, does dynamic
# orchestration beat flat parallel best-of-N?
#   mechanical = 100 isolated terra rollouts; report best-of-100 AND (+1 session) a sol reducer,
#                so the synthesis effect can be separated from the orchestration effect.
#   native     = sol orchestrator, 100-call budget, CDC scaffolding, terra subagents.
# Fixes vs the first run: multi-agent config now in config.toml (concurrency, model overrides,
# tool_namespace per openai/codex#31814), per-agent budgets enforced via mcp-session-id.
set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB/portfolio"
for PROB in fc46 fc302; do
  for ARM in mechanical native; do
    echo "########## $PROB / $ARM ##########"
    "$PY" -u mapreduce.py --problem $PROB --arm $ARM --n 100 --replicate 1 \
        --orch-style cdc --rollout-wall 1200 --orch-wall 5400 \
        --max-parallel 25 --grader-conc 32 --topk 5 || echo "^^ $PROB/$ARM FAILED"
  done
done
echo "########## EXP-1 v2 DONE ##########"
for d in "$REPO"/runs/mapreduce/*_r1; do
  [ -f "$d/result.json" ] && tr -d '\n' < "$d/result.json" && echo
done
