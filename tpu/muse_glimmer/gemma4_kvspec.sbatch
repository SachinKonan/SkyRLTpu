#!/bin/bash
# Regression check for the KV-cache-group change against the gemma-4 the repo
# actually runs in production (`tpu/runs/gemma4-31b.env` -> google/gemma-4-31B-it).
#
# This is the REAL config.json, not a fixture: the snapshot is already in the
# shared HF cache.  If the patched and baseline builds emit an identical KV
# cache spec for it, gemma-4-31b serving is unchanged *by construction*, which
# is a stronger statement than a sampled smoke could make.
#
# Reuses the two trees kv_spec_probe.sbatch laid down; run it after that job.
#
#SBATCH --job-name=mg-g4spec
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/g4spec-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
VENV=/n/fs/vision-mix/sk7524/caches/muse-parity/vllmvenv
WORK=/n/fs/vision-mix/sk7524/caches/muse-kvspec
OUT=$REPO/runs/muse_glimmer
G4=$(ls -d /n/fs/vision-mix/sk7524/caches/.hf/hub/models--google--gemma-4-31b-it/snapshots/*/ 2>/dev/null | head -1)

export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export HF_HOME=/n/fs/vision-mix/sk7524/caches/.hf
export VLLM_LOGGING_LEVEL=ERROR
export TOKENIZERS_PARALLELISM=false

echo "=== node $(hostname) job ${SLURM_JOB_ID:-manual} $(date -Is) ==="
echo "gemma-4-31b snapshot: ${G4}"
[ -n "$G4" ] || exit 2
python3 -c "
import json;c=json.load(open('${G4}/config.json'));t=c.get('text_config',c)
print('config:', {k:t.get(k) for k in ('num_hidden_layers','sliding_window','head_dim','global_head_dim','num_key_value_heads','num_global_key_value_heads')})"

for v in patched baseline; do
  echo
  echo "################################ ${v} ################################"
  export PYTHONPATH="$WORK/$v"
  timeout 1800 "${VENV}/bin/python" "${REPO}/tpu/muse_glimmer/kv_spec_probe.py" \
    --synthetic "${G4}config.json" --tp 4 --max-model-len 4096 \
    --out "${OUT}/kvspec-${v}-gemma4-31b-real.json" 2>&1 | grep -vE "^(WARNING|ERROR|W[0-9])" | tail -25
done

echo
echo "=== identical spec between patched and baseline? ==="
if diff -q "${OUT}/kvspec-patched-gemma4-31b-real.json" \
           "${OUT}/kvspec-baseline-gemma4-31b-real.json" >/dev/null 2>&1; then
  echo "GEMMA4-31B-SPEC-IDENTICAL"
else
  echo "GEMMA4-31B-SPEC-DIFFERS"
  diff "${OUT}/kvspec-patched-gemma4-31b-real.json" \
       "${OUT}/kvspec-baseline-gemma4-31b-real.json" | head -40
fi
echo "=== done $(date -Is) ==="
