#!/bin/bash
# Usage: drive_extract2.sh <ckpt> <root> <name> <only-patterns> <donefile-to-check...>
CK=$1; ROOT=$2; NAME=$3; ONLY=$4; shift 4
for i in $(seq 1 14); do
  timeout 660 ssh -o BatchMode=yes -n della-gh "cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved/research/fair_v2/agents/denoiser && timeout 590 /scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python extract_den2.py --ckpt $CK --root $ROOT --name $NAME --budget 540 --only '$ONLY' 2>&1 | grep --line-buffered -v Warn"
  done=1; for f in "$@"; do [ -f $ROOT/$NAME/$f ] || done=0; done
  if [ $done = 1 ]; then echo "DRIVER DONE"; exit 0; fi
  sleep 3
done
echo "DRIVER GAVE UP"
