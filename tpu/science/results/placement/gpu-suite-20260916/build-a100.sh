#!/bin/bash
set -euo pipefail
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-science-placement
export PATH="$PWD/.science/venv-cuda/bin:/usr/local/cuda-12.6/bin:$PATH"
export CUDA_HOME=/usr/local/cuda-12.6
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
cmake -S .science/xplace-a100 -B .science/xplace-a100/build-sm80 -G Ninja '-DCMAKE_CUDA_ARCHITECTURES=80;86' -DPYTHON_EXECUTABLE="$PWD/.science/venv-cuda/bin/python"
cmake --build .science/xplace-a100/build-sm80 --parallel 8
cmake --install .science/xplace-a100/build-sm80

touch tpu/science/results/placement/gpu-suite-20260916/a100-build-ready
