#!/bin/bash
set -euo pipefail
FCROOT=/n/fs/vision-mix/sk7524/SkyRLTpu-frontiercs/third_party/discover
PY=/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover/.venv-ttd-discover/bin/python
cd "$FCROOT"
export PYTHONPATH="$FCROOT:${PYTHONPATH:-}"
which g++; g++ --version | head -1
exec "$PY" -u -m examples.frontier_algo.selftest
