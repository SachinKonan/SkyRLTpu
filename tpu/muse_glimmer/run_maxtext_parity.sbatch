#!/bin/bash
#SBATCH --job-name=mg-parity
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/muse-parity/logs/parity-%j.out
#SBATCH --error=/n/fs/vision-mix/sk7524/muse-parity/logs/parity-%j.out
#
# Muse-Glimmer MaxText <-> HF numerical parity, CPU only.
#
# Two stages in two ISOLATED venvs (see build_parity_venvs.sbatch):
#   1. hfvenv  -- torch + transformers@main, dumps the reference and the weights
#                 under the exact HF parameter names from the released
#                 model.safetensors.index.json.
#   2. mtvenv  -- jax + the muse-glimmer MaxText clone, pushes those weights
#                 through MaxText's real PARAM_MAPPING/HOOK_FNS and compares.
#
# Neither venv touches the SkyRL project environment; the live RL sweep pins
# transformers and must not move.
set -euo pipefail

ROOT=/n/fs/vision-mix/sk7524/muse-parity
HERE=/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/muse_glimmer
DUMP=$ROOT/dump
export JAX_PLATFORMS=cpu
export XLA_FLAGS="--xla_force_host_platform_device_count=1"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_OFFLINE=1
mkdir -p "$DUMP"

echo "=== node $(hostname) job ${SLURM_JOB_ID:-none} ==="
date

echo
echo "############ STAGE 1: HF reference ############"
"$ROOT/hfvenv/bin/python" "$HERE/parity_hf_dump.py" --out "$DUMP"

echo
echo "############ STAGE 2: MaxText ############"
"$ROOT/mtvenv/bin/python" "$HERE/parity_maxtext_check.py" --dump "$DUMP" \
    --model_name muse-glimmer-tiny --atol 1e-4 --rtol 1e-3
rc=$?

echo
echo "############ STAGE 3: real to_maxtext CLI -> orbax -> from_pretrained ############"
rm -rf "$ROOT/convwork"; mkdir -p "$ROOT/convwork"
"$ROOT/mtvenv/bin/python" "$HERE/convert_smoke.py" --dump "$DUMP" --work "$ROOT/convwork" \
    --model_name muse-glimmer-tiny
conv_rc=$?

echo
echo "############ SIZING (30B, real training shape) ############"
XLA_FLAGS="--xla_force_host_platform_device_count=4" "$ROOT/mtvenv/bin/python" "$HERE/sizing.py" || true

date
echo "=== parity exit $rc / convert exit $conv_rc ==="
exit $(( rc + conv_rc ))
