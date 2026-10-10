#!/bin/bash
D=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved/research/fair_v2/agents/consistency
O=/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency/holdout
for mode in lco lbo; do
  for i in 1 2 3 4 5 6; do
    [ -f $O/$mode/cross.npz ] && break
    timeout 1500 ssh -o BatchMode=yes -n della-gh "cd $D && /scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python holdout.py --mode $mode --budget 400" 2>&1 | grep --line-buffered -v Warn
    echo "$mode run $i rc=${PIPESTATUS[0]}"
  done
done
