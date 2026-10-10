#!/bin/bash
# usage: gloop.sh <tag> <spec> [n]; reruns gdirect.py (cached, resumable) until it completes once
TAG=$1; SPEC=$2; N=${3:-6}
LOG=/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/gdirect_$TAG.log
for i in $(seq 1 $N); do
  timeout 900 ssh -o BatchMode=yes -n della-gh "cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved/research/fair_v2/agents/statistics && timeout 590 /scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python gdirect.py --tag $TAG --spec '$SPEC' --test" >> $LOG 2>&1
  if grep -q "mean over .* infam" $LOG; then echo finished; exit 0; fi
done
