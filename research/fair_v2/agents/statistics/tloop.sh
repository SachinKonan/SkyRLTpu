#!/bin/bash
LOG=/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/tshift.log
for i in $(seq 1 12); do
  timeout 900 ssh -o BatchMode=yes -n della-gh "cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved/research/fair_v2/agents/statistics && /scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python trainshift.py 300" >> $LOG 2>&1
  if grep -q "all done" $LOG; then echo finished; exit 0; fi
done
