#!/bin/bash
# farm_status.sh [--json]: per-pool inference farm status (health, lease owner, live throughput). Run on della-milan.
W=/scratch/gpfs/ZHUANGL/sk7524/SkyRL-Main-infer
set -a; . $W/.rq4/supervisors/discovery.env; set +a
cd $W && exec /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/python -m tpu.swarm.ray_train.farm_status "$@"
