#!/bin/bash
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:10:00
set -euo pipefail
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-circuit-300s-v5p32
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
exec /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-hybrid-inference-migration/.venv/bin/python -m tpu.science.placement_start_portfolio --queue /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-circuit-300s-v5p32/tpu/science/results/placement/abuplace-xplace-starts-20260921
