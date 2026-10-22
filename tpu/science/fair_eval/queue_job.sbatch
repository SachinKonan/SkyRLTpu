#!/bin/bash
# One queue-draining job: 2 A100-80GB on one node, single-node Ray, 1 GPU per task, 1 hour.
# Submit as an array with %1 so only one job runs at a time; each job continues the queue.
#SBATCH --job-name=momo-fair-queue
#SBATCH --partition=gpu-test
#SBATCH --qos=gpu-test
#SBATCH --account=zhuangl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=01:00:00
set -uo pipefail
F=/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval
W=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-momo-placement
QUEUE=${QUEUE:-$F/queue}
module load cudatoolkit/12.6 2>/dev/null || true
export CUDA_HOME=/usr/local/cuda-12.6 PATH=$F/env/cuda/bin:$PATH
export RAY_TMPDIR=/tmp/ray-$SLURM_JOB_ID RAY_DEDUP_LOGS=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv
lscpu | grep "Model name"
exec $F/env/cuda/bin/python -u $W/tpu/science/fair_eval/queue_runner.py --queue "$QUEUE" \
  --wall-seconds 3600 --gpus 2 --cpus 24
