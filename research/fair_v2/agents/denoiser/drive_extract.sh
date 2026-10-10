#!/bin/bash
# Usage: drive_extract.sh <ckpt> <root> <name> ; loops short della-gh runs until all sets are done.
CK=$1; ROOT=$2; NAME=${3:-den}
for i in $(seq 1 12); do
  timeout 660 ssh -o BatchMode=yes -n della-gh "cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-fair-universe-improved/research/fair_v2/agents/denoiser && timeout 590 /scratch/gpfs/ZHUANGL/sk7524/fair-env-gh/bin/python extract_den.py --ckpt $CK --root $ROOT --name $NAME --budget 520 2>&1 | grep -v Warn" 
  if [ -f $ROOT/$NAME/test.npz ]; then echo "DRIVER DONE"; exit 0; fi
  sleep 3
done
echo "DRIVER GAVE UP"
