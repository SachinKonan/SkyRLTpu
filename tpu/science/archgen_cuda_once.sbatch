#!/bin/bash
#SBATCH --job-name=placement-archgen-cuda
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --gres=gpu:a100:1
#SBATCH --time=04:00:00
#SBATCH --output=tpu/science/results/placement/archgen-cuda-once/slurm-%j.log
set -euo pipefail
cd /scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-science-placement
PLACEMENT_OUT="$PWD/tpu/science/results/placement/archgen-cuda-once"
PLACEMENT_ENV="$PWD/.science/venv-cuda"
PLACEMENT_XPLACE="$PWD/.science/xplace-cuda"
export PLACEMENT_OUT
trap 'rc=$?; echo "job_exit_code=$rc" >> "$PLACEMENT_OUT/job-status.txt"' EXIT
echo "stage=setup job=$SLURM_JOB_ID host=$(hostname)" > "$PLACEMENT_OUT/job-status.txt"
nvidia-smi
export CUDA_HOME=/usr/local/cuda-12.6
export PATH="$CUDA_HOME/bin:$PATH"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
uv --no-config venv --python .science/venv/bin/python "$PLACEMENT_ENV"
uv --no-config pip install --python "$PLACEMENT_ENV/bin/python" torch==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv --no-config pip install --python "$PLACEMENT_ENV/bin/python" 'numpy<2.3' scipy tqdm absl-py matplotlib 'cmake>=3.24,<4' ninja pandas numba cairocffi opencv-python-headless seaborn pulp igraph
export PATH="$PLACEMENT_ENV/bin:$PATH"
python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.cuda.get_device_name(0))'
python - <<'PY'
from pathlib import Path
import shutil
root=Path.cwd()
src=root/'.science/archgen'
dst=root/'.science/archgen-cuda-run'
if not dst.exists():shutil.copytree(src,dst)
patches=src/'support/xplace_patches'
xp=root/'.science/xplace-cuda'
shutil.copytree(patches,xp,dirs_exist_ok=True)
p=xp/'CMakeLists.txt'
s=p.read_text().replace('print(int(torch.cuda.is_available()))','print(int(torch.version.cuda is not None))')
p.write_text(s)
PY
echo 'stage=build_xplace' >> "$PLACEMENT_OUT/job-status.txt"
cmake -S "$PLACEMENT_XPLACE" -B "$PLACEMENT_XPLACE/build" -G Ninja -DCMAKE_CUDA_ARCHITECTURES=80 -DPYTHON_EXECUTABLE="$PLACEMENT_ENV/bin/python"
cmake --build "$PLACEMENT_XPLACE/build" --parallel 4
cmake --install "$PLACEMENT_XPLACE/build"
PYTHONPATH="$PLACEMENT_XPLACE" python -c 'import torch; from cpp_to_py.cpybin import dct_cuda, gpugr, io_parser; print("Xplace CUDA imports passed")'
uv --no-config pip freeze --python "$PLACEMENT_ENV/bin/python" > "$PLACEMENT_OUT/requirements.lock"
echo 'stage=candidate_3600_second_cap' >> "$PLACEMENT_OUT/job-status.txt"
python tpu/science/archgen_cuda_once.py --output "$PLACEMENT_OUT"
echo 'stage=complete' >> "$PLACEMENT_OUT/job-status.txt"
