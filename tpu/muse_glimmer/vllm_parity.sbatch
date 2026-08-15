#!/bin/bash
# CPU parity gate for the torch/vLLM Muse-Glimmer port. No chips, no QRs.
#
# Two venvs, because vllm 0.23.0 and `transformers @ main` cannot coexist:
#   hfvenv  : transformers 5.16.0.dev0 -> writes the tiny reference
#   mgvllm  : vllm 0.23.0 + torch cpu  -> builds the torch model and compares
#
# Usage:  sbatch vllm_parity.sbatch [tiny|real|both]
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH --job-name=mg-vllm-parity
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/vllm-parity-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
HFVENV=/n/fs/vision-mix/sk7524/muse-parity/hfvenv
MGVENV=/n/fs/vision-mix/sk7524/muse-vllm/venv
OUT="${REPO}/runs/muse_glimmer/vllm_parity"
MODE="${1:-tiny}"

mkdir -p "$OUT"
cd "$REPO"

export HF_HUB_OFFLINE=1
export HF_HOME=/n/fs/vision-mix/sk7524/caches/.hf
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error
export VLLM_LOGGING_LEVEL=WARNING
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"
# vllm must not try to touch a GPU or spin up its usage-stats reporter.
export VLLM_NO_USAGE_STATS=1
export VLLM_TARGET_DEVICE=cpu
export CUDA_VISIBLE_DEVICES=""
export MG_PARITY_PORT=$((29500 + RANDOM % 500))
# vLLM inspects the model class in a subprocess; tpu_inference is a source tree
# here, not a wheel, so it has to be importable from a clean interpreter.
export PYTHONPATH="${REPO}/third_party/tpu-inference${PYTHONPATH:+:${PYTHONPATH}}"

echo "=== node $(hostname)  job ${SLURM_JOB_ID:-manual}  $(date -Is) ==="
rc=0

if [[ "$MODE" == "tiny" || "$MODE" == "both" ]]; then
  for variant in tiny gqa; do
    echo
    echo "############ reference: ${variant} (hfvenv) ############"
    "${HFVENV}/bin/python" "${REPO}/tpu/muse_glimmer/vllm_parity_ref.py" \
      --out-dir "$OUT" --variant "$variant" || { rc=2; continue; }

    echo
    echo "############ torch model: ${variant} (mgvllm) ############"
    "${MGVENV}/bin/python" "${REPO}/tpu/muse_glimmer/vllm_parity_check.py" \
      --mode tiny --variant "$variant" --ref-dir "$OUT" \
      --out "${OUT}/${variant}_result.json"
    [[ $? -ne 0 ]] && rc=3
  done

  echo
  echo "############ LoRA adapter count (mgvllm, CPU) ############"
  "${MGVENV}/bin/python" "${REPO}/tpu/muse_glimmer/vllm_parity_check.py" \
    --mode lora --variant tiny --ref-dir "$OUT" \
    --out "${OUT}/lora_result.json"
  [[ $? -ne 0 ]] && rc=5
fi

if [[ "$MODE" == "real" || "$MODE" == "both" ]]; then
  echo
  echo "############ real weights vs recorded HF dump (mgvllm) ############"
  "${MGVENV}/bin/python" "${REPO}/tpu/muse_glimmer/vllm_parity_check.py" \
    --mode real \
    --hf-ref "${REPO}/runs/muse_glimmer/hf_ref.npz" \
    --out "${OUT}/real_result.json" \
    "${@:2}"
  [[ $? -ne 0 ]] && rc=4
fi

echo
echo "=== vllm parity exit ${rc}  $(date -Is) ==="
exit $rc
