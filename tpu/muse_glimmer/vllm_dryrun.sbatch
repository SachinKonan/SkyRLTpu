#!/bin/bash
# Build a throwaway CPU venv holding vllm 0.23.0 + transformers@main + the LOCAL
# tpu-inference, and dry-run everything vLLM does before it touches a TPU.
# Catches the integration failures (config parse, arch registration, weight
# mapping) on a CPU node instead of on a 3-hour spot slice.
#
# Isolated on purpose: the project venv pins transformers<=5.8.0 and vllm==0.23.0
# for the live RL sweep and must not move.
#
#SBATCH --job-name=mg-dryrun
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=03:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/dryrun-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
VENV=/n/fs/vision-mix/sk7524/caches/muse-parity/vllmvenv
MODEL=/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b
export UV_CACHE_DIR=/n/fs/vision-mix/sk7524/caches/.uv
export HF_HOME=/n/fs/vision-mix/sk7524/caches/.hf
# CRITICAL: slurm runs this with cwd = the repo root, and the repo's
# pyproject.toml carries `[tool.uv] override-dependencies = [... "transformers
# >=5.6.1,<=5.8.0" ...]` for the live RL sweep.  uv applies that OVERRIDE even to
# an explicit `transformers @ git+...@main` requirement and even with
# --force-reinstall --no-deps, so the install silently yields 5.8.0 and every
# muse_glimmer stage then fails with "Transformers does not recognize this
# architecture".  Discovering no config at all is the fix.
export UV_NO_CONFIG=1
mkdir -p "${REPO}/runs/muse_glimmer"

echo "=== node $(hostname) job ${SLURM_JOB_ID:-manual} $(date -Is) ==="

if [[ ! -x "${VENV}/bin/python" || "${MG_REBUILD_VENV:-0}" == "1" ]]; then
  echo "=== building ${VENV} ==="
  rm -rf "${VENV}"
  uv venv --python 3.12 "${VENV}" || exit 2
  # vllm first (pulls its own torch), then override transformers and overlay
  # the local tpu-inference -- exactly the layering the TPU host will use.
  uv pip install --python "${VENV}/bin/python" "vllm==0.23.0" 2>&1 | tail -5 || exit 2
  uv pip install --python "${VENV}/bin/python" \
    "jax[cpu]>=0.8" "flax>=0.12.2" "safetensors" 2>&1 | tail -3 || exit 2
  # torchax ships inside the vllm-tpu wheel the TPU host installs, but NOT in
  # the plain vllm wheel used here.  tpu_inference.layers.vllm.backends imports
  # it at module scope, so without it the general_plugins entry point dies with
  # ModuleNotFoundError, vLLM SWALLOWS that, and the arch never registers --
  # which then cascades into stages 3 and 6.  Install it so the CPU dry-run
  # exercises the same plugin path the TPU host will.
  uv pip install --python "${VENV}/bin/python" "torchax" "qwix" "jaxtyping" 2>&1 | tail -3 || exit 2
  uv pip install --python "${VENV}/bin/python" --no-deps --force-reinstall \
    "${REPO}/third_party/tpu-inference" 2>&1 | tail -3 || exit 2
  # transformers@main goes LAST and with --no-deps.  Any later resolving
  # install (the jax/flax step used to be one) re-resolves the environment and
  # silently DOWNGRADES transformers back to 5.8.0 to satisfy vllm's pin --
  # which is exactly what made the first dry-run fail stages 1/2/3/6.
  uv pip install --python "${VENV}/bin/python" --no-deps --force-reinstall \
    "transformers @ git+https://github.com/huggingface/transformers@main" 2>&1 | tail -3 || exit 2
else
  echo "=== reusing ${VENV}; refreshing the tpu-inference overlay ==="
  uv pip install --python "${VENV}/bin/python" --no-deps --force-reinstall \
    "${REPO}/third_party/tpu-inference" 2>&1 | tail -3
  for mod in torchax qwix jaxtyping; do
    if ! "${VENV}/bin/python" -c "import ${mod}" >/dev/null 2>&1; then
      echo "=== ${mod} missing (tpu-inference imports it) -- installing ==="
      uv pip install --python "${VENV}/bin/python" "${mod}" 2>&1 | tail -3 || exit 2
    fi
  done
  # Repair a clobbered transformers in place (cheap) rather than rebuilding.
  tv=$("${VENV}/bin/python" -c 'import transformers;print(transformers.__version__)' 2>/dev/null)
  echo "transformers currently ${tv}"
  case "$tv" in
    5.[0-9].*|5.1[0-4].*|4.*|"")
      echo "=== transformers ${tv} is too old for muse_glimmer -- reinstalling @main ==="
      uv pip install --python "${VENV}/bin/python" --no-deps --force-reinstall \
        "transformers @ git+https://github.com/huggingface/transformers@main" 2>&1 | tail -3 || exit 2
      ;;
  esac
fi

"${VENV}/bin/python" -c 'import vllm, transformers, jax, flax; print("vllm", vllm.__version__, "| transformers", transformers.__version__, "| jax", jax.__version__)' || exit 3

export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=4"
export VLLM_TARGET_DEVICE=cpu
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
# Leave VLLM_PLUGINS unset so ALL general plugins load (setting it to a subset
# would silently skip tpu-inference's register_layers entry point).
unset VLLM_PLUGINS

echo
"${VENV}/bin/python" "${REPO}/tpu/muse_glimmer/vllm_dryrun.py" \
  --model-dir "$MODEL" --layers "${MG_LAYERS:-4}" --tp "${MG_TP:-4}" \
  --out "${REPO}/runs/muse_glimmer/dryrun.json" 2>&1
rc=$?
echo "=== dryrun exit ${rc} $(date -Is) ==="
exit $rc
