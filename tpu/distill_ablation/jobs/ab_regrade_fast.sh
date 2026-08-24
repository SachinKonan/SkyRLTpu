#!/bin/bash
# Recover the fast arm's real production numbers (its server ran --no-full, blocking the driver's
# yardstick). Programs are on disk; NO codex spend.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
cd "$AB/portfolio"
exec "$PY" -u regrade_arm.py --run-dir "$REPO/runs/ab_scheme/erdos" --arm fast --problem erdos --group 10
