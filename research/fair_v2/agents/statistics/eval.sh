#!/bin/bash
# usage: eval.sh <outname> <models> <combos> [extra args]; runs fit_eval.py on della-gh (blocking, <10 min)
OUT=/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/runs/$1
mkdir -p $OUT
timeout 900 ssh -o BatchMode=yes -n della-gh "cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved/research/fair_v2/agents/statistics && /scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python fit_eval.py --out $OUT --models '$2' --combos '$3' --reuse $4" 2>&1 | grep -v "nll step\|Warning\|warn" | tee -a $OUT/eval.log
