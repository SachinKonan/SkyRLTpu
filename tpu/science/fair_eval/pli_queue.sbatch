#!/bin/bash
# One queue-draining job on a PLI H100 node: 2 H100, 48 cores (4 slices of 12), 256 GB, 1 hour.
# The runner packs tasks onto this pool: a GPU-bound placer takes most of one GPU plus a 12-core slice,
# a CPU-bound placer (MOMO) takes a 12-core slice and a sliver of a GPU for its short Xplace step.
# Many such jobs drain the same queue concurrently (--array=0-N%K); each keeps its resources for the
# hour and keeps starting tasks that fit.
#SBATCH --job-name=momo-fair-pliq
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=48
#SBATCH --mem=256G
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
exec $F/env/cuda/bin/python -u $W/tpu/science/fair_eval/queue_runner.py --queue "$QUEUE" --wall-seconds 3600 --slice-cpus 12
