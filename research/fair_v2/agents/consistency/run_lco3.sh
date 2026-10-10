#!/bin/bash
# usage: run_lco3.sh <set>
K=$1
D=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved/research/fair_v2/agents/consistency
O=/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency/lco3
PY=/scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python
FAMS=dog,cnn,den_r,N2,stx_pkcl,PsBh,small,large,b0,b1,b2,b3,cnn1,den_sd,psL,psS,onept,extrema,mf
for spec in "std:$FAMS" "cj:dog,cnn,den_r" "cj2:dog,cnn,den_r" "sm:dog,cnn,den_r"; do
  V=${spec%%:*}; F=${spec#*:}
  for i in 1 2 3 4 5 6; do
    [ -f $O/set${K}_$V/cross.npz ] && break
    n=$(echo $F | tr ',' '\n' | while read f; do [ -f $O/set${K}_$V/$f.npz ] || echo $f; done | wc -l)
    [ "$n" = "0" ] && [ "$V" != "std" ] && break
    timeout 1500 ssh -o BatchMode=yes -n della-gh "cd $D && $PY holdout3.py --set $K --variant $V --fams $F --budget 420" 2>&1 | grep --line-buffered -v Warn
  done
done
echo "set $K all-done"
