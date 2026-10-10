#!/usr/bin/env bash
# Invoke inside a single-chip systemd unit on the authorized lab host.
set -euo pipefail
variant="${1:?morph, regional or joint}"
case "$variant" in morph|regional|joint) ;; *) exit 2 ;; esac
labroot=/dev/shm/fair-r3-20261009
labpython=/home/gcpuser/.cache/inference-farm-rq4-qwen35-east5a-v5p32-008/envs/serving/bin/python
test -f "$labroot/features-${variant}/manifest.json"
for seed in 0 1 2; do
  "$labpython" "$labroot/code/fair_r2/flow.py" fit --raw /dev/shm/fair-r2-20261009/raw --features "$labroot/features-${variant}" --out "$labroot/flow-${variant}-s${seed}" --probes "$labroot/features-${variant}/probes" --condition cosmology --seed "$seed" --steps 10000 --batch 256 --layers 8 --width 128 --floor 0.0001 --transform asinh --scale-bound 0.5
done
