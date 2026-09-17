#!/bin/bash
set -euo pipefail
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-science-placement
export CUDA_VISIBLE_DEVICES=0 PLACEMENT_GPU_NODE=/dev/nvidia0
export OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for case in ibm01 ibm04 ibm08 ibm18; do
 .science/venv-cuda/bin/python tpu/science/placement_gpu_suite.py --method xplace --case "$case" --output "tpu/science/results/placement/gpu-suite-20260916/xplace-$case" --repository "$PWD/.science/archgen-cuda-run" --xplace-root "$PWD/.science/xplace-cuda" --seconds 1100 || exit $?
done
