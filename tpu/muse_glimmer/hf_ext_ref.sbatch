#!/bin/bash
# Extra HF references for the follow-up TPU run: decode across the 2048
# sliding-window boundary, and long-context (8k / 16k) teacher-forced argmax.
#
# CPU only.  30B in float32 is ~112 GB resident, plus a [16384, 202048] float32
# logit tensor (13.2 GB), so this asks for a lot of RAM and never runs on the
# login node.
#
#SBATCH --job-name=mg-extref
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=280G
#SBATCH --time=06:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/extref-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
VENV=/n/fs/vision-mix/sk7524/caches/muse-parity/venv
MODEL=/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b
OUT="${REPO}/runs/muse_glimmer"
mkdir -p "$OUT"

export HF_HUB_OFFLINE=1
export HF_HOME=/n/fs/vision-mix/sk7524/caches/.hf
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"

echo "=== node $(hostname) job ${SLURM_JOB_ID:-manual} $(date -Is) ==="

# Free part first: the dense window-boundary probe set comes straight out of
# the existing teacher-forced argmax, no model needed.
"${VENV}/bin/python" "${REPO}/tpu/muse_glimmer/make_ext_ref.py" \
  --npz "${OUT}/hf_ref.npz" \
  --out-json "${OUT}/mg_ext_ref.json" \
  --out-npz "${OUT}/mg_logit_rows.npz" 2>&1

echo
echo "############## HF decode-crossing + long context ##############"
"${VENV}/bin/python" "${REPO}/tpu/muse_glimmer/hf_ext_ref.py" \
  --model-dir "$MODEL" \
  --ref-npz "${OUT}/hf_ref.npz" \
  --out "${OUT}/hf_ext_ref.json" \
  --attn sdpa 2>&1
rc=$?
echo "=== hf_ext_ref exit ${rc} $(date -Is) ==="
exit $rc
