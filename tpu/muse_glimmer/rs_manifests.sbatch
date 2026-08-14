#!/bin/bash
# Build the three reasoning-strength manifests (erdos, jssp, ac1) on CPU.
#
# Scope of this round: 4 parent states + 1 start state per problem, 32
# completions, 2 arms  ->  320 items/problem, 960 total.
#
# Runs on a compute node because the acineq step-15 pool is ~400 MB of JSON
# and the tokenizer load is not free.
#
#SBATCH --job-name=mg-rsman
#SBATCH --partition=all
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH --output=/n/fs/vision-mix/sk7524/slurm-logs/mg-rsman-%j.log

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
WORK=/n/fs/vision-mix/sk7524/muse-rs2
PY="$REPO/third_party/discover/.venv/bin/python"
export MUSE_GLIMMER_DIR=/n/fs/vision-mix/sk7524/caches/muse-glimmer-30b
export PYTHONPATH="$REPO/third_party/discover"

echo "=== node $(hostname) $(date -Is) ==="
mkdir -p "$WORK/manifests"

# 1. the client's thread pool really does run N requests at once. This is the
#    single fact the previous attempt could not see until 1012 s in, so it is
#    proven on CPU against a mock server before any slice time is spent.
"$PY" "$REPO/tpu/muse_glimmer/rs_conc_selftest.py" || exit 1

for p in erdos jssp ac1; do
  echo "--- $p ---"
  "$PY" "$REPO/tpu/muse_glimmer/rs_build_manifest.py" \
    --problem "$p" \
    --pool-dir "$WORK/pools" \
    --out "$WORK/manifests/$p.json" \
    --n-parents "${NP:-4}" --n-seeds "${NS:-1}" \
    --rollouts "${ROLL:-32}" --start-rollouts "${SROLL:-32}" \
    --seed "${SEED:-20260813}" --current-date "${CDATE:-2026-08-13}" \
    --log-path "$WORK/envlog"
  echo "  exit $?"
done
ls -la "$WORK/manifests"
echo "=== done $(date -Is) ==="
