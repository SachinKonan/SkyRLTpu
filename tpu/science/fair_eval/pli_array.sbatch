#!/bin/bash
# One fair-eval task per array element on PLI H100 nodes (8 GPUs / 96 cores / 1 TB per node):
# 1 H100, 12 cores (the per-GPU share), 64 GB (the official eval container cap), 1 hour. Submit with --array=0-(N-1)%CONCURRENCY
# and PLAN=<plan dir> (contains index.txt and specs/).
#SBATCH --job-name=momo-fair-pli
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=01:00:00
set -uo pipefail
F=/scratch/gpfs/ZHUANGL/sk7524/placement-fair-eval
W=/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-momo-placement
module load cudatoolkit/12.6 2>/dev/null || true
export CUDA_HOME=/usr/local/cuda-12.6
exec $F/env/cuda/bin/python -u $W/tpu/science/fair_eval/run_spec.py --plan "$PLAN" --index "$SLURM_ARRAY_TASK_ID"
