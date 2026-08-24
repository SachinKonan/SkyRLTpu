#!/bin/bash
# 1) regrade the fast arm (its --no-full server blocked the driver's yardstick), 2) audit BOTH arms
# for hardcoded / returns-base programs so the A/B table reports GENUINE results only. No codex.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
cd "$AB/portfolio"
"$PY" -u regrade_arm.py --run-dir "$REPO/runs/ab_scheme/erdos" --arm fast --problem erdos --group 10
"$PY" -u audit_programs.py --run-dir "$REPO/runs/ab_scheme/erdos" --arms fast full
