#!/usr/bin/env bash
# Erdős multi-seed merge round 2: fork rebut-merge-erdos-{qwen,muse}-{a,b} at aggregated step 12 with the merged
# step-12 tree (see tpu/merge_seeds_setup.py). Dry run by default.
#   bash tpu/merge_seeds_setup.sh [--arm qwen-a ...] [--prepare | --go]
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/gpfs/ZHUANGL/sk7524/.cache/uv}
exec uv run --isolated --no-project --python 3.12 --with numpy python tpu/merge_seeds_setup.py "$@"
