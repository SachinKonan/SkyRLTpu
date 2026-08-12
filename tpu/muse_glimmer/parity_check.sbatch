#!/bin/bash
# CPU-only parity gate for the JAX Muse-Glimmer text port.
# No chips, no QRs -- pure CPU numerics.
#
# The HF side needs `muse_glimmer`, which only exists in unreleased
# transformers (>= 5.16.0.dev0).  Upgrading the project environment would break
# the live RL sweep, so the harness runs from a dedicated throwaway venv that
# holds nothing but torch(cpu) + jax(cpu) + transformers@main.  Build it once
# (on a node with network) with:
#
#   uv venv --python 3.12 "$MG_PARITY_VENV"
#   VIRTUAL_ENV="$MG_PARITY_VENV" uv pip install --torch-backend cpu \
#       torch numpy 'jax[cpu]' \
#       'transformers @ git+https://github.com/huggingface/transformers@main'
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 0:45:00
#SBATCH --job-name=mg-parity
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/parity-%j.log

set -uo pipefail
repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
venv="${MG_PARITY_VENV:-/n/fs/vision-mix/sk7524/caches/muse-parity/venv}"
mkdir -p "${repo_root}/runs/muse_glimmer"
cd "$repo_root"

export JAX_PLATFORMS=cpu
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export HF_HUB_OFFLINE=1          # nothing here needs the hub
export TRANSFORMERS_VERBOSITY=error

if [[ ! -x "${venv}/bin/python" ]]; then
  echo "FATAL: parity venv not found at ${venv} (see header for how to build it)"
  exit 2
fi

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-manual}  $(date -Is) ==="
"${venv}/bin/python" -c 'import transformers, torch, jax; print("transformers", transformers.__version__, "| torch", torch.__version__, "| jax", jax.__version__)'

"${venv}/bin/python" "${repo_root}/tpu/muse_glimmer/parity_check.py" "$@"
rc=$?

echo
echo "=== parity_check.py exit ${rc} ==="
exit $rc
