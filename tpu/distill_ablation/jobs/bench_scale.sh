#!/bin/bash
# SCALING experiment: same CDC prompt, but reasoning=max + 48 subagents (up from high/32).
# Full procedure = fresh run (Stage 1) THEN continuation loop (Stage 2), matching the original
# so the comparison is apples-to-apples. Writes to <problem>_scale tags so the original runs
# (erdos=0.380906, fc302=0.437404) stay intact. Strictly serial: one codex session live at a time.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB"

PROBS="erdos fc302"
COMMON="--reasoning max --max-agents 48 --max-concurrent 24 --tag-suffix _scale --backend thread"

echo "[scale] $(date) STAGE 1 — fresh runs ($PROBS) at reasoning=max, 48 subagents"
"$PY" -u benchmark_cdc.py --problems $PROBS $COMMON

echo "[scale] $(date) STAGE 2 — continuation loop ($PROBS) at reasoning=max, 48 subagents"
"$PY" -u continue_loop.py --problems $PROBS $COMMON \
    --iter-wall 1800 --patience 2 --max-iters 10

echo "[scale] $(date) DONE"
