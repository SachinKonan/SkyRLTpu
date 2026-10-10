#!/usr/bin/env bash
# Run only inside a single-chip service on the authorized lab host.
set -euo pipefail
labroot=/dev/shm/fair-r3-20261009
labpython=/home/gcpuser/.cache/inference-farm-rq4-qwen35-east5a-v5p32-008/envs/serving/bin/python
test -f "$labroot/noise-views/manifest.json"
mkdir -p "$labroot/probes72"
for probe in development r1_novel_now_reused; do
  test -f "/dev/shm/fair-r2-20261009/probes72/${probe}.npz"
  cp "/dev/shm/fair-r2-20261009/probes72/${probe}.npz" "$labroot/probes72/"
done
cp /dev/shm/fair-r2-20261009/morphology-probes/features72/morphology_r2.npz "$labroot/probes72/"
for seed in 0 1 2; do
  "$labpython" "$labroot/code/fair_r3/fit_noise_views.py" \
    --raw /dev/shm/fair-r2-20261009/raw \
    --features /dev/shm/fair-r2-20261009/features72 \
    --views "$labroot/noise-views" --out "$labroot/flow-noise4-s${seed}" \
    --probes "$labroot/probes72" --seed "$seed" --steps 10000 \
    --batch 256 --layers 8 --width 128 --floor 0.0001 --transform asinh --scale-bound 0.5
done
