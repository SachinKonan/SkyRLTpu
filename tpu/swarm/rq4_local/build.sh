#!/bin/bash
# Live RQ4 runs and farms build from the frozen release worktree SkyRL-Main-infer-release (agent/rq4-release).
# Build (and optionally upload) a profile from SkyRL-Main-infer: build.sh <run_id> [--upload]
W=/scratch/gpfs/ZHUANGL/sk7524/SkyRL-Main-infer-release
cd "$W"
run=$1; shift
exec /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/python -m tpu.swarm.ray_train.build "tpu/swarm/ray_train/profiles/${run}.json" --output ".rq4/builds/${run}" "$@"
