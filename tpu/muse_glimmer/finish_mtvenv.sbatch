#!/bin/bash
#SBATCH --job-name=mg-mtvenv
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/muse-parity/logs/mtvenv-%j.out
#SBATCH --error=/n/fs/vision-mix/sk7524/muse-parity/logs/mtvenv-%j.out
#
# MaxText declares `dependencies = []` (all deps live in optional extras), so the
# import graph has to be satisfied by hand. Rather than guess the closure, import
# maxtext in a loop and install whatever module it complains about next.
set -uo pipefail

ROOT=/n/fs/vision-mix/sk7524/muse-parity
PY="$ROOT/mtvenv/bin/python"
export UV_CACHE_DIR=${UV_CACHE_DIR:-/n/fs/vision-mix/sk7524/caches/.uv}
export JAX_PLATFORMS=cpu
cd "$ROOT"

# module name -> pip name, where they differ
declare -A PIPNAME=(
  [pathwaysutils]=pathwaysutils
  [tensorboardX]=tensorboardX
  [google]=google-cloud-storage
  [ml_goodput_measurement]=ml-goodput-measurement
  [cloud_accelerator_diagnostics]=cloud-accelerator-diagnostics
  [grain]=grain
  [tokamax]=tokamax
  [aqt]=aqtp
  [jaxtyping]=jaxtyping
  [ml_collections]=ml-collections
  [PIL]=pillow
  [yaml]=pyyaml
  [tensorflow]=tensorflow-cpu
  [datasets]=datasets
  [tiktoken]=tiktoken
  [sentencepiece]=sentencepiece
  [orbax]=orbax-checkpoint
  [etils]="etils[epath]"
  [jsonlines]=jsonlines
  [tqdm]=tqdm
  [psutil]=psutil
  [numpy]=numpy
  [einops]=einops
  [safetensors]=safetensors
  [huggingface_hub]=huggingface-hub
  [transformers]=transformers
  [optax]=optax
  [chex]=chex
  [flax]=flax
  [absl]=absl-py
  [pydantic]=pydantic
  [omegaconf]=omegaconf
  [google_cloud_mldiagnostics]=google-cloud-mldiagnostics
  [drjax]=drjax
  [qwix]=qwix
  [clu]=clu
  [array_record]=array-record
)

for attempt in $(seq 1 40); do
  echo "=== import attempt $attempt ==="
  out=$("$PY" -c '
import maxtext, jax, flax
from maxtext.common.common_types import DecoderBlockType
from maxtext.models import muse_glimmer
import maxtext.configs.pyconfig as pyconfig
from maxtext.checkpoint_conversion.utils.param_mapping import PARAM_MAPPING
from maxtext.checkpoint_conversion.to_maxtext import get_maxtext_model_info
print("IMPORT_OK", jax.__version__, flax.__version__, DecoderBlockType.MUSE_GLIMMER,
      muse_glimmer.MUSE_GLIMMER_ATTENTION_PATTERN, "muse-glimmer-30b" in PARAM_MAPPING)
' 2>&1)
  echo "$out" | tail -20
  if echo "$out" | grep -q IMPORT_OK; then
    echo "=== IMPORTS SATISFIED ==="
    break
  fi
  mod=$(echo "$out" | sed -n "s/.*No module named '\([A-Za-z0-9_.]*\)'.*/\1/p" | tail -1 | cut -d. -f1)
  if [[ -z "$mod" ]]; then
    echo "=== not a missing-module error; stopping ==="
    break
  fi
  pkg=${PIPNAME[$mod]:-$mod}
  echo "--- installing $pkg (for module $mod) ---"
  VIRTUAL_ENV="$ROOT/mtvenv" uv pip install --python "$PY" "$pkg" 2>&1 | tail -3
done

"$PY" -m pip list 2>/dev/null | head -80 || true
echo "=== DONE ==="
