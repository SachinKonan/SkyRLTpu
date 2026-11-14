#!/usr/bin/env bash
# Run A snapshot ensemble: fork a base run @ batch 11 into its Run A arm (see tpu/runA_setup.py; arms muse-s3,
# muse-s2, qwen-s3). Dry run by default.
#   bash tpu/runA_setup.sh --arm <arm> [--prepare | --go]
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/gpfs/ZHUANGL/sk7524/.cache/uv}
exec uv run --isolated --no-project --python 3.12 --with numpy python tpu/runA_setup.py "$@"
