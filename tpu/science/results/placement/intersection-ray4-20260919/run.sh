#!/bin/bash
#SBATCH --partition=gpu-test
#SBATCH --qos=gpu-test
#SBATCH --account=zhuangl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --signal=B:TERM@45
set -euo pipefail
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-science-placement
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD"
OUT="$PWD/tpu/science/results/placement/intersection-ray4-20260919"
# Slurm's scheduled end includes elapsed setup; do not reset the hour here.
END=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | sed -n 's/^EndTime=//p')
DEADLINE=$(date -d "$END" +%s)
nvidia-smi --query-gpu=name,memory.total --format=csv
exec .science/venv-gpu-ray/bin/python -m tpu.science.gpu_ray_queue \
    --queue "$OUT" --deadline "$DEADLINE" --gpus 4 --cpus-per-task 8 --task-seconds 3300
