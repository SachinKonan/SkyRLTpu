#!/bin/bash
# One queue-draining job on a PLI H100 node: 1 H100, 36 cores (3 slices of 12), 192 GB, 1 hour.
# queue_runner.py packs tasks onto this pool: a GPU-bound placer (ArchGen, AbuPlace) takes most of
# the GPU and a 12-core slice; a CPU-bound placer (MOMO) takes a 12-core slice and a sliver of the
# GPU for its short Xplace step. One GPU per three slices matches the work mix. Many such jobs drain
# the same queue concurrently (--array=0-N%K); each keeps its resources for the hour and keeps
# starting tasks that fit.
#SBATCH --job-name=momo-fair-pliq
#SBATCH --partition=pli
#SBATCH --account=llm_explore
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=36
#SBATCH --mem=192G
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
lscpu -p=CPU,CORE,SOCKET,NODE | grep -v "^#" | awk -F, -v aff="$(taskset -pc $$ | cut -d: -f2)" 'BEGIN{print "allowed cpus:" aff}' 
exec $F/env/cuda/bin/python -u $W/tpu/science/fair_eval/queue_runner.py --queue "$QUEUE" --wall-seconds 3600 --slice-cpus 12
