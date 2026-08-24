#!/bin/bash
# Stage-2 DECOUPLED codegen: terra implements each plan in read-only NO-RUN mode
# (hard no-tune guarantee), caches solution.py to workdirs. No grading here — that's
# a separate parallel regrade phase. AB_PLANS (file in corpora/), AB_TAG (scratch tag).
set -euo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu; AB=$REPO/tpu/distill_ablation
PY="${TTD_AB_VENV}/bin/python"
CTRL=$REPO/runs/ttd_gptoss20b_ctrl15/tinker_log/erdos-gptoss20b-ctrl15/puct_sampler_step_000015.json
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cd "$AB"
exec "$PY" -u run_executor.py --plans "$AB/corpora/${AB_PLANS:?}" \
  --executor-model gpt-5.6-terra --exec-mode norun --no-grade \
  --tag "${AB_TAG:?}" --exec-concurrency 8 --exec-timeout 240 \
  --worse-set "$AB/corpora/worse_set.json" --pool-snapshot "$CTRL" \
  --out "$AB/corpora/${AB_TAG}_codegen.json"
