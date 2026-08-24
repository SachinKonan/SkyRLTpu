#!/bin/bash
# Smoke: build a tiny self corpus (paid, small), then finetune+eval on 2 seeds x 2
# rollouts with short generations. Confirms the whole loop end-to-end for ~$0.15.
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
HO=$AB/heldout_seeds.json
cd "$AB"

echo "=== [smoke] build tiny corpus (target ~6 survivors) ==="
"$PY" -u build_corpus.py --mode self --name smoke \
  --snapshot "$CTRL" --heldout "$HO" \
  --max-pairs 12 --teacher-phase1-tokens 26000 --max-target-tokens 4096 --chunk 12 \
  --out "$AB/corpora/smoke.json"

echo "=== [smoke] finetune (4 datums, 3 epochs) + eval (2 seeds x 2 rollouts, short) ==="
"$PY" -u finetune_and_eval.py --arm SMOKE \
  --corpus "$AB/corpora/smoke.json" --subset-size 4 --epochs 3 \
  --probe "$AB/corpora/smoke.json" \
  --k 2 --max-seeds 2 --phase1-max-tokens 12000 \
  --heldout "$HO" --eval-pool-snapshot "$CTRL" \
  --out-dir "$REPO/runs/distill_ablation/SMOKE"

echo "=== [smoke] done; summary ==="
cat "$REPO/runs/distill_ablation/SMOKE/summary.json"
