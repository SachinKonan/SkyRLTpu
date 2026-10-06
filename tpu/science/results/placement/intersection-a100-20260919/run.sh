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
OUT="$PWD/tpu/science/results/placement/intersection-a100-20260919"
test -f "$OUT/build-ready"
read -r METHOD CASE < <(python - "$SLURM_ARRAY_TASK_ID" "$OUT/plan.json" <<'PY'
import json,sys
row=json.load(open(sys.argv[2]))['jobs'][int(sys.argv[1])]
print(row['method'],row['case'])
PY
)
if [[ "$METHOD" == abuplace ]]; then REPO="$PWD/.science/abuplace"; else REPO="$PWD/.science/archgen-cuda-run"; fi
# Use a private runtime copy: Xplace writes logs/cache beneath its own tree.
XP="$OUT/runtime-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir "$XP"
cp -a .science/xplace-a100-intersection-20260919/{src,utils,cpp_to_py,data,thirdparty,tool,main.py,CMakeLists.txt} "$XP/"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=16 MKL_NUM_THREADS=1
# Slurm's device cgroup bounds access. UUID avoids index remapping in bwrap.
GPU_UUID=$(.science/venv-cuda/bin/python - <<'PY'
import torch
assert torch.cuda.device_count()==1
assert "A100" in torch.cuda.get_device_name(0), torch.cuda.get_device_name(0)
print("GPU-" + str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-"))
PY
)
export CUDA_VISIBLE_DEVICES="$GPU_UUID"
nvidia-smi --query-gpu=name,memory.total --format=csv
exec .science/venv-cuda/bin/python tpu/science/placement_gpu_suite.py --method "$METHOD" --case "$CASE" --repository "$REPO" --xplace-root "$XP" --output "$OUT/$METHOD-$CASE" --seconds 3450
