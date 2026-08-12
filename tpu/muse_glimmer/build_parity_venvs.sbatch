#!/bin/bash
#SBATCH --job-name=mg-venvs
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/n/fs/vision-mix/sk7524/muse-parity/logs/venvs-%j.out
#SBATCH --error=/n/fs/vision-mix/sk7524/muse-parity/logs/venvs-%j.out
#
# Build the two ISOLATED CPU venvs the Muse-Glimmer parity harness needs.
# Neither touches the SkyRL project environment (the live RL sweep pins
# transformers/vllm and must not move).
#
#   hfvenv : torch (CPU) + transformers @ git main  -- MuseGlimmer ships in
#            transformers 5.15.0.dev0, unreleased.
#   mtvenv : jax (CPU) + flax/orbax + the muse-glimmer MaxText clone (editable).
set -euo pipefail

ROOT=/n/fs/vision-mix/sk7524/muse-parity
MAXTEXT=/n/fs/vision-mix/sk7524/maxtext-muse
export UV_CACHE_DIR=${UV_CACHE_DIR:-/n/fs/vision-mix/sk7524/caches/.uv}
export HF_HUB_OFFLINE=0
mkdir -p "$ROOT/logs"
cd "$ROOT"

echo "=== node: $(hostname)  job: ${SLURM_JOB_ID:-none} ==="
date

# ---------------------------------------------------------------- HF venv ----
echo "=== building hfvenv (torch cpu + transformers@main) ==="
uv venv --python 3.12 "$ROOT/hfvenv"
VIRTUAL_ENV="$ROOT/hfvenv" uv pip install --python "$ROOT/hfvenv/bin/python" \
  --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  "torch" "numpy" "safetensors" "accelerate" \
  "transformers @ git+https://github.com/huggingface/transformers@main"

"$ROOT/hfvenv/bin/python" - <<'PY'
import transformers, torch
print("transformers", transformers.__version__, "torch", torch.__version__)
from transformers.models.muse_glimmer import modeling_muse_glimmer as m
print("muse_glimmer OK:", [n for n in dir(m) if n.startswith("MuseGlimmerText")][:8])
PY

# ------------------------------------------------------------ MaxText venv ---
echo "=== building mtvenv (jax cpu + maxtext clone) ==="
uv venv --python 3.12 "$ROOT/mtvenv"
VIRTUAL_ENV="$ROOT/mtvenv" uv pip install --python "$ROOT/mtvenv/bin/python" \
  "jax[cpu]" "flax" "orbax-checkpoint" "optax" "numpy" "pydantic" "pyyaml" \
  "omegaconf" "aqtp" "safetensors" "absl-py" "ml-collections" "tensorstore" \
  "jaxtyping" "einops" "chex" "cloudpickle" "etils[epath]" "annotated-doc" \
  "transformers==4.57.1" "sentencepiece" "tiktoken" "huggingface-hub"

# MaxText itself: no runtime deps declared (`dependencies = []`), so install the
# clone without deps and let the hand-picked list above satisfy it.
VIRTUAL_ENV="$ROOT/mtvenv" uv pip install --python "$ROOT/mtvenv/bin/python" \
  --no-deps -e "$MAXTEXT"

"$ROOT/mtvenv/bin/python" - <<'PY'
import jax, flax, maxtext
print("jax", jax.__version__, "flax", flax.__version__)
print("maxtext at", maxtext.__file__)
from maxtext.common.common_types import DecoderBlockType
print("MUSE_GLIMMER enum:", DecoderBlockType.MUSE_GLIMMER)
from maxtext.models import muse_glimmer
print("pattern:", muse_glimmer.MUSE_GLIMMER_ATTENTION_PATTERN)
PY

echo "=== DONE ==="
date
