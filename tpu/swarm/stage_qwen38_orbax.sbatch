#!/usr/bin/env bash
#SBATCH --job-name=qwen38-orbax
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=320G
#SBATCH --time=08:00:00
set -euo pipefail
: "${SLURM_JOB_ID:?Submit using sbatch}"
: "${QWEN38_STAGE_ROOT:?Explicit dedicated stage root required}"

revision=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
snapshot="$QWEN38_STAGE_ROOT/hf/hub/models--Qwen--Qwen3.8-27B/snapshots/$revision"
converter="$QWEN38_STAGE_ROOT/converter-venv/bin/python"
output="$QWEN38_STAGE_ROOT/orbax/qwen3.5-27b"
test -s "$QWEN38_STAGE_ROOT/download-complete.json"
test -s "$snapshot/model.safetensors.index.json"
test ! -e "$output/CHECKPOINT_COMPLETE" || exit 0
# A partial output requires explicit inspection; never erase another attempt.
test ! -d "$output/0" || { echo 'Partial checkpoint exists; inspect before retrying' >&2; exit 1; }
mkdir -p "$output" "$QWEN38_STAGE_ROOT/config-cwd"
export JAX_PLATFORMS=cpu HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export HF_HOME="$QWEN38_STAGE_ROOT/hf"
export XLA_FLAGS=--xla_force_host_platform_device_count=16
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=16 OPENBLAS_NUM_THREADS=1

configs_dir=$("$converter" -c 'from pathlib import Path; import maxtext; print(Path(maxtext.__file__).parent / "configs")')
ln -sfn "$configs_dir" "$QWEN38_STAGE_ROOT/config-cwd/src"
cd "$QWEN38_STAGE_ROOT/config-cwd"
"$converter" -m maxtext.checkpoint_conversion.to_maxtext \
  model_name=qwen3.5-27b \
  base_output_directory="$output" \
  tokenizer_path="$snapshot" \
  scan_layers=True use_multimodal=false skip_jax_distributed_system=True \
  checkpoint_storage_use_ocdbt=True checkpoint_storage_use_zarr3=True \
  --lazy_load_tensors=True --hf_model_path="$snapshot" --revision="$revision" \
  --simulated_cpu_devices_count=16

test -s "$output/0/items/_METADATA"
test -s "$output/0/items/manifest.ocdbt"
"$converter" "$SLURM_SUBMIT_DIR/tpu/swarm/validate_qwen38_orbax.py" "$output" "$snapshot" "$revision"
