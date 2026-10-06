#!/bin/bash
#SBATCH --partition=gpu-test
#SBATCH --qos=gpu-test
#SBATCH --account=zhuangl
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=01:00:00
set -euo pipefail
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-science-placement
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$PWD/.science/venv-cuda/bin:$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
XP="$PWD/.science/xplace-a100-intersection-20260919"
nvidia-smi --query-gpu=name,memory.total --format=csv
cmake -S "$XP" -B "$XP/build-sm80" -G Ninja -DCMAKE_CUDA_ARCHITECTURES=80 -DPYTHON_EXECUTABLE="$PWD/.science/venv-cuda/bin/python"
cmake --build "$XP/build-sm80" --parallel 8
cmake --install "$XP/build-sm80"
PYTHONPATH="$XP" python -c 'import torch; from cpp_to_py.cpybin import dct_cuda, gpugr, io_parser; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0))'
"$CUDA_HOME/bin/cuobjdump" --list-elf "$XP"/cpp_to_py/cpybin/dct_cuda*.so
 touch tpu/science/results/placement/intersection-a100-20260919/build-ready
