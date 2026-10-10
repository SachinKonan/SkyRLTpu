#!/usr/bin/env bash
# Frozen pointwise inference only, on an authorized erdos-3242 lab host.
set -euo pipefail
seed="${1:?provide 0 or 1}"
case "$seed" in 0|1) ;; *) exit 2 ;; esac
labroot=/dev/shm/fair-r2-20261009
labpython=/home/gcpuser/.cache/inference-farm-rq4-qwen35-east5a-v5p32-008/envs/serving/bin/python
test -f "$labroot/field-s${seed}/complete.json"
test -f "$labroot/hybrid-test-cache/manifest.json"
if sudo fuser /dev/vfio/* >/dev/null 2>&1; then
  printf '%s\n' 'TPU owner still present; nothing launched.' >&2
  exit 3
fi
for chip in 0 1 2 3; do
  test ! -e "$labroot/field-test-s${seed}-${chip}"
done
for chip in 0 1 2 3; do
  sudo systemd-run --unit="fair-r2-field-test-s${seed}-${chip}" --uid=gcpuser --collect --property=LimitMEMLOCK=infinity --property=MemoryMax=80G --property=MemorySwapMax=0 --setenv=TPU_VISIBLE_CHIPS="$chip" --setenv=TPU_CHIPS_PER_HOST_BOUNDS=1,1,1 --setenv=TPU_HOST_BOUNDS=1,1,1 --setenv=OPENBLAS_NUM_THREADS=2 --setenv=OMP_NUM_THREADS=2 --setenv=PYTHONUNBUFFERED=1 "$labpython" "$labroot/code/fair_r2/field_benchmark.py" score --run "$labroot/field-s${seed}" --cache "$labroot/hybrid-test-cache" --out "$labroot/field-test-s${seed}-${chip}" --batch 32 --ode-steps 32 --shard "$chip" --shards 4
done
