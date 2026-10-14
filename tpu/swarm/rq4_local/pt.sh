#!/bin/bash
# Run pytest in the SkyRL-Main-infer checkout, importing its own discover copy.
W=/scratch/gpfs/ZHUANGL/sk7524/SkyRL-Main-infer
cd "$W"
export PYTHONPATH="$W:$W/third_party/discover"
exec /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-remote-only-trainer/.venv/bin/python -m pytest -q -p no:cacheprovider "$@"
