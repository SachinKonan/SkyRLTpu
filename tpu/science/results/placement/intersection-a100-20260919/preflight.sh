#!/bin/bash
#SBATCH --partition=gpu-test
#SBATCH --qos=gpu-test
#SBATCH --account=zhuangl
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:05:00
set -euo pipefail
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-science-placement
export PYTHONPATH="$PWD/.science/xplace-a100-intersection-20260919"
export LD_LIBRARY_PATH="$PWD/.science/xplace-a100-intersection-20260919/cpp_to_py/cpybin"
.science/venv-cuda/bin/python - <<'PY'
import torch
from cpp_to_py.cpybin import dct_cuda,gpugr,io_parser,draw_placement
assert torch.cuda.device_count()==1
assert 'A100' in torch.cuda.get_device_name(0)
x=torch.ones((32,32),device='cuda');assert x.sum().item()==1024
print('A100_RUNTIME_IMPORTS_PASS',torch.cuda.get_device_name(0),torch.__version__,flush=True)
PY
