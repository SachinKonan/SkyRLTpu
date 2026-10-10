#!/bin/bash
# score newly added shift sets with the frozen family models (+ fit the new cnn1 family), then the scale cross-evaluations
D=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved/research/fair_v2/agents/consistency
PY=/scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python
for i in 1 2 3; do
  timeout 1500 ssh -o BatchMode=yes -n della-gh "cd $D && $PY fit_families.py --mode families --only psL,psS,onept,extrema,mf,dog,cnn,small,mid,large,cnn1 --budget 420" 2>&1 | grep --line-buffered -v "Warn\|nll step"
done
for i in 1 2 3; do
  timeout 1500 ssh -o BatchMode=yes -n della-gh "cd $D && timeout 560 $PY scalecons.py --extra --tag scale2" 2>&1 | grep -v Warn | tr '\n' ' '
  echo " scale2 run $i"
done
echo update-done
