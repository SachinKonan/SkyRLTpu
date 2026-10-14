#!/bin/bash
# Launch a built profile onto a SkyPilot pool: launch.sh <run_id> <pool>
export SKYPILOT_DISABLE_LOCAL_API_SERVER=1
run=$1; pool=$2
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRL-Main-infer-release
exec /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/sky jobs launch ".rq4/builds/${run}/${run}.yaml" --pool "$pool" --yes --detach-run
