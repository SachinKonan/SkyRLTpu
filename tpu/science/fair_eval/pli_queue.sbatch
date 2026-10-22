#!/bin/bash
# One queue-draining job on a PLI H100 node: 2 H100, 24 cores (12 per GPU, the node's per-GPU share),
# 128 GB (64 GB per task), 1 hour. Many such jobs drain the same queue concurrently
# (submit as --array=0-N%K). Each keeps its GPUs for the hour and keeps starting tasks that fit.
#SBATCH --job-name=momo-fair-pliq
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=128G
#SBATCH --time=01:00:00
set -uo pipefail
F=/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval
W=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-momo-placement
QUEUE=${QUEUE:-$F/pli-queue}
module load cudatoolkit/12.6 2>/dev/null || true
export CUDA_HOME=/usr/local/cuda-12.6 PATH=$F/env/cuda/bin:$PATH
export RAY_TMPDIR=/tmp/ray-$SLURM_JOB_ID RAY_DEDUP_LOGS=0
nvidia-smi --query-gpu=index,name,memory.total --format=csv
lscpu | grep "Model name"
exec $F/env/cuda/bin/python -u $W/tpu/science/fair_eval/queue_runner.py --queue "$QUEUE" --wall-seconds 3600
