#!/usr/bin/env bash
#SBATCH --job-name=gptoss120b-orbax
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# The patched converter fills scanned tensors incrementally instead of
# retaining and restacking all 18 layers. The final tensor plus conversion
# buffers exceeded 160 GiB during asynchronous Orbax serialization, so retain
# headroom above that measured peak.
#SBATCH --mem=192G
#SBATCH --time=1-00:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/gptoss120b-orbax-stage-%j.log
#SBATCH --error=/n/fs/vision-mix/sk7524/gptoss120b-orbax-stage-%j.log
#
# One stable CPU job converts the immutable BF16 source once and publishes the
# Orbax checkpoint. TPU pool setup requires CHECKPOINT_COMPLETE, written last,
# so no TPU host can consume a torn upload.
set -euo pipefail

: "${SLURM_JOB_ID:?submit this file with sbatch}"

# Slurm executes a spool copy of this file, so BASH_SOURCE does not point back
# into the checkout. SLURM_SUBMIT_DIR is the durable repository location.
REPO="${SKYRL_REPO_DIR:-${SLURM_SUBMIT_DIR:?Slurm submit directory is required}}"
test -f "$REPO/tpu/tunix_runtime/pyproject.toml" || {
  echo "SkyRL checkout not found at $REPO" >&2
  exit 2
}
STAGE_ROOT="${GPTOSS120B_STAGE_ROOT:-/n/fs/vision-mix/sk7524/gptoss120b-orbax-stage}"
MAXTEXT_COMMIT="${GPTOSS120B_MAXTEXT_COMMIT:-5e916f853b8f095466bf8e49537a88e4c11fafb7}"
HF_MODEL="${GPTOSS120B_HF_MODEL:-unsloth/gpt-oss-120b-BF16}"
HF_REVISION="${GPTOSS120B_HF_REVISION:-e7523373bc44b42296b43202e265a1eebf2ee16f}"
GCS_ROOT="${TUNIX_MAXTEXT_CKPT_CACHE_GCS:-gs://sk7524-tinker-tpu-asia-northeast1/skyrl-maxtext-ckpts-gptoss120b-bf16-d388}"
GCS_TARGET="$GCS_ROOT/gpt-oss-120b"
OUTPUT_ROOT="$STAGE_ROOT/output/gpt-oss-120b"
MARKER="$OUTPUT_ROOT/CHECKPOINT_COMPLETE"
VENV="$STAGE_ROOT/venv"

mkdir -p "$STAGE_ROOT" "$STAGE_ROOT/output" "$STAGE_ROOT/huggingface"
case "$OUTPUT_ROOT" in
  */output/gpt-oss-120b) ;;
  *) echo "refusing unsafe stage output path: $OUTPUT_ROOT" >&2; exit 2 ;;
esac
if gcloud storage objects describe "$GCS_TARGET/CHECKPOINT_COMPLETE" \
  >/dev/null 2>&1; then
  echo "GPT-OSS 120B checkpoint is already published at $GCS_TARGET"
  exit 0
fi

export HF_HOME="$STAGE_ROOT/huggingface"
export HF_XET_HIGH_PERFORMANCE=1
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=16"
export PYTHONUNBUFFERED=1
export UV_CACHE_DIR="${UV_CACHE_DIR:-/n/fs/vision-mix/sk7524/caches/.uv}"
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

UV_PROJECT_ENVIRONMENT="$VENV" uv sync \
  --project "$REPO/tpu/tunix_runtime" --frozen
uv pip install --python "$VENV/bin/python" \
  "maxtext @ git+https://github.com/SachinKonan/maxtext.git@$MAXTEXT_COMMIT" \
  'drjax>=0.1.4' aqtp pathwaysutils tokamax tiktoken

valid_local_checkpoint() {
  [ -s "$OUTPUT_ROOT/0/_CHECKPOINT_METADATA" ] && \
    [ -s "$OUTPUT_ROOT/0/items/_METADATA" ] && \
    [ -s "$OUTPUT_ROOT/0/items/_sharding" ] && \
    [ -s "$OUTPUT_ROOT/0/items/manifest.ocdbt" ]
}

if ! valid_local_checkpoint; then
  # Orbax does not resume a partially-written OCDBT tree. This path is scoped
  # to the dedicated stage root and is safe to rebuild after a failed CPU job.
  rm -rf -- "$OUTPUT_ROOT"
  mkdir -p "$OUTPUT_ROOT"
  configs_dir=$("$VENV/bin/python" - <<'PY'
from pathlib import Path
import maxtext
print(Path(maxtext.__file__).parent / "configs")
PY
)
  config_cwd="$STAGE_ROOT/maxtext-config"
  mkdir -p "$config_cwd"
  ln -sfn "$configs_dir" "$config_cwd/src"
  (
    cd "$config_cwd"
    "$VENV/bin/python" -m maxtext.checkpoint_conversion.to_maxtext \
      model_name=gpt-oss-120b \
      base_output_directory="$OUTPUT_ROOT" \
      scan_layers=True \
      use_multimodal=false \
      skip_jax_distributed_system=True \
      checkpoint_storage_use_ocdbt=True \
      checkpoint_storage_use_zarr3=True \
      --lazy_load_tensors=True \
      --hf_model_path="$HF_MODEL" \
      --revision="$HF_REVISION" \
      --simulated_cpu_devices_count=16
  )
fi

valid_local_checkpoint || {
  echo "conversion exited without a complete Orbax layout at $OUTPUT_ROOT" >&2
  exit 1
}
checkpoint_bytes=$(du -sb "$OUTPUT_ROOT" | awk '{print $1}')
if [ "$checkpoint_bytes" -lt 50000000000 ]; then
  echo "checkpoint is implausibly small: $checkpoint_bytes bytes" >&2
  exit 1
fi

{
  printf 'model=%s\n' 'gpt-oss-120b'
  printf 'hf_model=%s\n' "$HF_MODEL"
  printf 'hf_revision=%s\n' "$HF_REVISION"
  printf 'maxtext_commit=%s\n' "$MAXTEXT_COMMIT"
  printf 'checkpoint_bytes=%s\n' "$checkpoint_bytes"
  printf 'built_at=%s\n' "$(date -u +%FT%TZ)"
} > "$MARKER"

gcloud storage rsync -r --exclude='(^|/)CHECKPOINT_COMPLETE$' \
  "$OUTPUT_ROOT" "$GCS_TARGET"
gcloud storage cp "$MARKER" "$GCS_TARGET/CHECKPOINT_COMPLETE"
gcloud storage objects describe "$GCS_TARGET/CHECKPOINT_COMPLETE" >/dev/null
echo "GPT-OSS 120B Orbax checkpoint published: $GCS_TARGET ($checkpoint_bytes bytes)"
