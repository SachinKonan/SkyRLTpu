#!/bin/bash
# Real-weight CPU parity for the torch/vLLM Muse-Glimmer port.
#
# 27.855 B parameters in float32 is ~112 GB resident -- a compute node, never
# the login node.  The HF reference is NOT recomputed: runs/muse_glimmer/
# hf_ref.npz already holds the teacher-forced argmax, top-8 sets and sampled
# full logit rows for the five E2E prompts, produced by the same
# transformers@main build the JAX port was gated against.  So only one 30B copy
# is ever live.
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH -t 6:00:00
#SBATCH --job-name=mg-vllm-real
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/vllm-real-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
MGVENV=/n/fs/vision-mix/sk7524/muse-vllm/venv
MODEL=/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b
OUT="${REPO}/runs/muse_glimmer/vllm_parity"
mkdir -p "$OUT"
cd "$REPO"

export HF_HUB_OFFLINE=1
export HF_HOME=/n/fs/vision-mix/sk7524/caches/.hf
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=WARNING
export VLLM_NO_USAGE_STATS=1
export CUDA_VISIBLE_DEVICES=""
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export MG_PARITY_PORT=$((29500 + RANDOM % 500))
export PYTHONPATH="${REPO}/third_party/tpu-inference${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-manual}  $(date -Is) ==="
free -g | head -2

"${MGVENV}/bin/python" "${REPO}/tpu/muse_glimmer/vllm_parity_check.py" \
  --mode real \
  --model-dir "$MODEL" \
  --hf-ref "${REPO}/runs/muse_glimmer/hf_ref.npz" \
  --max-model-len 8192 \
  --out "${OUT}/real_result.json" \
  "$@"
rc=$?

echo
echo "=== real-weight vllm parity exit ${rc}  $(date -Is) ==="
exit $rc
