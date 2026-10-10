#!/bin/bash
# usage: loop.sh <kind> <max_iters>; reruns extract_fam.py on della-gh in <10 min pieces until done
KIND=$1; N=${2:-12}
LOG=/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/extract_${KIND}.log
for i in $(seq 1 $N); do
  timeout 900 ssh -o BatchMode=yes -n della-gh "cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved/research/fair_v2/agents/statistics && /scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python extract_fam.py --kind $KIND --test --budget 450" >> $LOG 2>&1
  if tail -1 $LOG | grep -q "^done in"; then
    if ! grep -q "skip shift_" <(tail -30 $LOG); then echo "finished $KIND"; exit 0; fi
    sleep 120
  fi
done
