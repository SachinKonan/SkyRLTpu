#!/bin/bash
# Floorplan-test every method's placement on the given designs (4 in parallel per design), then summarize.
O=/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval/orfs
T=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-momo-placement/tpu/science/fair_eval/orfs
cd /scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval
export PATH=$O/fptest/bin:$PATH
for d in "$@"; do
  for m in momofill momo300 archgen abuplace; do
    timeout 1500 nice env/cuda/bin/python $T/floorplan_test.py $d $m > $O/fptest/$m-$d.log 2>&1 &
  done
  wait
done
for d in "$@"; do
  for m in momofill momo300 archgen abuplace; do
    b=$O/fptest/$m-$d/ORFS/flow
    L=$(ls -d $b/logs/nangate45/*/base | head -1)
    if ls $(ls -d $b/results/nangate45/*/base | head -1)/2_4_floorplan_pdn.odb >/dev/null 2>&1; then pdn=OK; else pdn=$(grep -h ERROR $L/2_*.log 2>/dev/null | head -1); fi
    echo "$d $m | $(grep -h 'Placed .* macros' $L/2_2_floorplan_macro.log 2>/dev/null) | $(grep -ho 'macros moved (mean [^,]*' $O/fptest/$m-$d.log) | $pdn"
  done
done
