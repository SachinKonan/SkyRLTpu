#!/bin/bash
# run_local.sh <run_id>: start one trainer-less no-train run in the background on this host.
set -euo pipefail
run=$1
L=/scratch/gpfs/ZHUANGL/sk7524/rq4-local
root=$L/runs/$run
profile=$root/pkg/tpu/swarm/ray_train/profiles/$run.json
model=$(python3 -c "import json,sys; print(json.load(open(sys.argv[1]))['model_preset'])" "$profile")
case $model in
  qwen3.5-27b) key=models--Qwen--Qwen3.5-27B;;
  gemma4-31b) key=models--google--gemma-4-31B-it;;
  muse-glimmer-30b) key=models--meta-models--Muse-Glimmer-30B;;
esac
snapshot=$L/hf/hub/$key/snapshots/$(cat $L/hf/hub/$key/refs/main)
# Eager frozen steps (end a step at 95% graded; late grades land a step later) only for listed runs.
# Circuit's grading tail is hours long, so its steps release at 90% (user, 2026-10-13).
if grep -qx "$run" $L/scripts/eager_runs.txt 2>/dev/null; then
  case "$run" in *-circuit-*) export TTD_EAGER_STEP_FRACTION=0.90;; *) export TTD_EAGER_STEP_FRACTION=0.95;; esac
fi
# Muse prompts carry the date; a run listed here keeps the date its bootstrap contract was written with.
d=$(awk -v r="$run" '$1==r {print $2}' $L/scripts/renderer_dates.txt 2>/dev/null); [ -n "$d" ] && export TTD_RENDERER_DATE=$d
cd "$root/pkg"
PYTHONPATH="$root/pkg" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 nohup $L/envs/controller/bin/python -u -m tpu.swarm.ray_train.local_notrain \
  --config "$profile" --registry $L/farms.json --package "$root/pkg" --source "$(readlink -f $root/source)" \
  --python $L/envs/client/bin/python --hf-home $L/hf --snapshot "$snapshot" --num-epochs ${NUM_EPOCHS:-25} ${FARM_GROUP_CAP:+--farm-group-cap $FARM_GROUP_CAP} ${FARM_TARGET:+--farm-target $FARM_TARGET} > "$root/runner.log" 2>&1 &
echo $! > "$root/runner.pid"
echo "$run pid=$(cat $root/runner.pid)"
