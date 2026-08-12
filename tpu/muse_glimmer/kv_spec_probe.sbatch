#!/bin/bash
# A/B the KV-cache-group change on CPU, before any slice is spent.
#
# Two copies of the tpu-inference tree are made: one with the working-tree edit
# (PATCHED) and one with `git show HEAD:...` of kv_cache_manager.py written back
# over it (BASELINE).  Each is put first on PYTHONPATH, so the same venv runs
# both without reinstalling anything and without touching the real submodule
# working tree (the TPU run rsyncs from it concurrently).
#
# For each: the repo's own kv_cache_manager unit tests, then the group-count
# probe for muse-glimmer and for the gemma-4 variants.
#
#SBATCH --job-name=mg-kvspec
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=01:30:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/muse_glimmer/kvspec-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
SUB=$REPO/third_party/tpu-inference
VENV=/n/fs/vision-mix/sk7524/caches/muse-parity/vllmvenv
MODEL=/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b
OUT=$REPO/runs/muse_glimmer
WORK=/n/fs/vision-mix/sk7524/caches/muse-kvspec
mkdir -p "$OUT"

export JAX_PLATFORMS=cpu
export HF_HUB_OFFLINE=1
export HF_HOME=/n/fs/vision-mix/sk7524/caches/.hf
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false

echo "=== node $(hostname) job ${SLURM_JOB_ID:-manual} $(date -Is) ==="

# The dry-run venv was never built with a test runner; the repo's own
# kv_cache_manager tests are the sharpest available statement about what the
# change does to models other than muse-glimmer, so it is worth the 20 seconds.
"${VENV}/bin/python" -c 'import pytest' 2>/dev/null || \
  uv pip install --python "${VENV}/bin/python" pytest 2>&1 | tail -2

rm -rf "$WORK"; mkdir -p "$WORK"
for v in patched baseline; do
  rsync -a --exclude '.git' --exclude '__pycache__' "$SUB/" "$WORK/$v/"
done
git -C "$SUB" show HEAD:tpu_inference/runner/kv_cache_manager.py \
  > "$WORK/baseline/tpu_inference/runner/kv_cache_manager.py" || exit 2
echo "patched vs baseline differ:"
diff -q "$WORK/patched/tpu_inference/runner/kv_cache_manager.py" \
        "$WORK/baseline/tpu_inference/runner/kv_cache_manager.py" || true

for v in patched baseline; do
  echo
  echo "################################ ${v} ################################"
  export PYTHONPATH="$WORK/$v"
  echo "--- unit tests: tests/runner/test_kv_cache_manager.py ---"
  (cd "$WORK/$v" && timeout 1800 "${VENV}/bin/python" -m pytest -q \
      tests/runner/test_kv_cache_manager.py 2>&1 | tail -25)
  for s in muse-glimmer gemma4-e2b gemma4-31b; do
    echo "--- probe (synthetic ${s}) ---"
    timeout 900 "${VENV}/bin/python" "${REPO}/tpu/muse_glimmer/kv_spec_probe.py" \
      --synthetic "$s" --tp 4 --out "${OUT}/kvspec-${v}-${s}.json" 2>&1 | tail -20
  done
  echo "--- probe (REAL muse-glimmer config) ---"
  timeout 900 "${VENV}/bin/python" "${REPO}/tpu/muse_glimmer/kv_spec_probe.py" \
    --model-dir "$MODEL" --tp 4 --out "${OUT}/kvspec-${v}-real.json" 2>&1 | tail -25
done

echo "=== done $(date -Is) ==="
