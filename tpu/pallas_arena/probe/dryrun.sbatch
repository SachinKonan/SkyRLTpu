#!/bin/bash
# CPU dry run for the cold-start probe: every moving part except the chips.
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 0:50:00
#SBATCH --job-name=pallas-probe-dryrun
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/probe-dryrun-%j.log

set -uo pipefail
repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$repo_root"
export PYTHONPATH="${repo_root}/tpu:${PYTHONPATH:-}"
export JAX_PLATFORMS=cpu

echo "=== observation unit battery ==="
uv run --isolated --extra dev --extra jax --with flatbuffers python -m pytest \
  tpu/pallas_arena/tests/test_observation.py -q -p no:cacheprovider
rc1=$?

echo "=== probe dry run ==="
uv run --isolated --extra dev --extra jax --with flatbuffers python tpu/pallas_arena/probe/dryrun.py
rc2=$?

echo "=== legacy arena battery (regression: the probe must not have broken it) ==="
uv run --isolated --extra dev --extra jax --with flatbuffers python -m pytest \
  tpu/pallas_arena/tests/ -q -p no:cacheprovider
rc3=$?

echo "=== dryrun rc: observation=${rc1} probe=${rc2} legacy=${rc3} ==="
exit $(( rc1 | rc2 | rc3 ))
