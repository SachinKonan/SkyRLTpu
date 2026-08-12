#!/bin/bash
# Real-weight teacher-forced parity for the Muse-Glimmer text port.
#
# Runs on a CPU compute node (30B in float32 is ~119 GB resident -- never the
# login node).  Two sequential processes so only one 30B copy is live at a
# time: the torch reference dumps logits/hidden-states/greedy ids, then the
# JAX core is fed the same safetensors and compared against that dump.
#
#SBATCH --job-name=mg-real
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=250G
#SBATCH --time=08:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/real-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
VENV=/n/fs/vision-mix/sk7524/caches/muse-parity/venv
MODEL=/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b
OUT="${REPO}/runs/muse_glimmer"
mkdir -p "$OUT"

export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export HF_HOME=/n/fs/vision-mix/sk7524/caches/.hf
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "=== node $(hostname) job ${SLURM_JOB_ID:-manual} $(date -Is) ==="
"${VENV}/bin/python" -c 'import transformers, torch, jax; print("transformers", transformers.__version__, "| torch", torch.__version__, "| jax", jax.__version__)'

echo
echo "############## SIDE: hf (torch fp32 reference) ##############"
/usr/bin/time -v "${VENV}/bin/python" "${REPO}/tpu/muse_glimmer/real_weight_check.py" \
  --side hf --model-dir "$MODEL" --out "${OUT}/hf_ref.npz" 2>&1
rc_hf=$?
echo "=== hf side exit ${rc_hf} $(date -Is) ==="
if [[ $rc_hf -ne 0 ]]; then echo "ABORT: hf side failed"; exit $rc_hf; fi

echo
echo "############## SIDE: jax (muse_glimmer_core fp32) ##############"
/usr/bin/time -v "${VENV}/bin/python" "${REPO}/tpu/muse_glimmer/real_weight_check.py" \
  --side jax --model-dir "$MODEL" --ref "${OUT}/hf_ref.npz" \
  --out "${OUT}/jax_vs_hf.json" 2>&1
rc_jax=$?
echo "=== jax side exit ${rc_jax} $(date -Is) ==="
exit $rc_jax
