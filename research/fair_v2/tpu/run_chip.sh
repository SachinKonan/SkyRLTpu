#!/bin/bash
# run_chip.sh <chip> <name> [jax_unet args...]: one training process pinned to one TPU chip.
chip=$1; name=$2; shift 2
export TPU_VISIBLE_CHIPS=$chip TPU_PROCESS_BOUNDS=1,1,1 TPU_CHIPS_PER_PROCESS_BOUNDS=1,1,1 \
       TPU_PROCESS_PORT=$((8476 + chip)) TPU_PROCESS_ADDRESSES=localhost:$((8476 + chip)) CLOUD_TPU_TASK_ID=0 \
       JAX_PLATFORMS=tpu XLA_PYTHON_CLIENT_PREALLOCATE=true
unset JAX_COORDINATOR_ADDRESS TPU_MULTIHOST_BACKEND
mkdir -p /dev/shm/fair/runs/$name
exec /dev/shm/fair/venv/bin/python /dev/shm/fair/code/jax_unet.py --out /dev/shm/fair/runs/$name "$@" > /dev/shm/fair/runs/$name/train.log 2>&1
