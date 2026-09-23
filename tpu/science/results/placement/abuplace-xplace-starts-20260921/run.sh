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
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-circuit-300s-v5p32
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
export PYTHONPATH="$PWD"
END=$(scontrol show job "$SLURM_JOB_ID" -o | tr ' ' '\n' | sed -n 's/^EndTime=//p')
DEADLINE=$(date -d "$END" +%s)
exec /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-science-placement/.science/venv-gpu-ray/bin/python -m tpu.science.gpu_ray_queue --queue /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-circuit-300s-v5p32/tpu/science/results/placement/abuplace-xplace-starts-20260921 --deadline "$DEADLINE" --gpus 4 --cpus-per-task 8 --task-seconds 1980
