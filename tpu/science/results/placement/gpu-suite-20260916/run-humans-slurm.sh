#!/bin/bash
set -euo pipefail
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-science-placement
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cases=(ibm04 ibm08 ibm18)
idx=${SLURM_ARRAY_TASK_ID}
if ((idx < 3)); then method=archgen; repo="$PWD/.science/archgen-cuda-run"; else method=abuplace; repo="$PWD/.science/abuplace"; fi
case=${cases[$((idx % 3))]}
nvidia-smi --query-gpu=name,memory.total --format=csv
exec .science/venv-cuda/bin/python tpu/science/placement_gpu_suite.py --method "$method" --case "$case" --repository "$repo" --xplace-root "$PWD/.science/xplace-a100" --output "tpu/science/results/placement/gpu-suite-20260916/$method-$case" --seconds 3450
