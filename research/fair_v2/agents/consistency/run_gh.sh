#!/bin/bash
# usage: run_gh.sh <mode> [only] ; loops short della-gh runs until every output exists
MODE=$1; ONLY=$2
D=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved/research/fair_v2/agents/consistency
for i in 1 2 3 4 5 6; do
  timeout 1500 ssh -o BatchMode=yes -n della-gh "cd $D && /scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python fit_families.py --mode $MODE --only '$ONLY' --budget 400" 2>&1 | grep -v Warning
  rc=${PIPESTATUS[0]}
  echo "run $i rc=$rc"
  out=$(timeout 300 ssh -o BatchMode=yes -n della-gh "cd $D && /scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python -c \"import lib,sys; from pathlib import Path; m='$MODE'; o=lib.OUT/('fam' if m=='families' else 'r1box'); names=list(lib.FAMILIES) if m=='families' else ['Sd','C','S','SC']; only='$ONLY'; names=[n for n in names if not only or n in only.split(',')]; print(sum(not (o/(n+'.npz')).exists() for n in names))\"")
  echo "remaining: $out"
  [ "$out" = "0" ] && break
done
