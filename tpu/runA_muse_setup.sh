#!/usr/bin/env bash
# Run A snapshot ensemble, Muse seed 3: fork rebut-erdos-muse-seed3 @ batch 11 for the v5p-16 trainer
# (K/V 8 -> 2 with Adam, minimal registry, one-row ledger, step-11 tree). Dry run by default.
#   bash tpu/runA_muse_setup.sh [--prepare | --go]
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR=${UV_CACHE_DIR:-/scratch/gpfs/ZHUANGL/sk7524/.cache/uv}
exec uv run --isolated --no-project --python 3.12 --with numpy python tpu/runA_muse_setup.py "$@"
