#!/bin/bash
# Phase 0 baseline-importability gate for the Pallas kernel arena.
# COMPUTE NODES ONLY — uv env resolution + jax imports are real work.
#
# Submit:  sbatch tpu/pallas_arena/run_phase0.sbatch
# Logs:    runs/pallas_arena/phase0-<jobid>.log (+ phase0-results.jsonl)
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH -t 1:00:00
#SBATCH --job-name=pallas-arena-phase0
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/phase0-%j.log

set -uo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: run under SLURM (sbatch); no compute on the login node." >&2
  exit 1
fi

repo_root=/n/fs/vision-mix/sk7524/SkyRLTpu
out="${repo_root}/runs/pallas_arena/phase0-results.jsonl"
scratch="${SLURM_TMPDIR:-/tmp}/pallas-arena-phase0-${SLURM_JOB_ID}"
mkdir -p "$scratch"
rm -f "$out"

cd "$repo_root"
overall=0

echo "=== stage main (dev+tunix env) $(date +%H:%M:%S) ==="
if ! uv run --isolated --extra dev --extra tunix \
    python tpu/pallas_arena/phase0_import_gate.py \
    --stage main --out "$out" --scratch "$scratch"; then
  echo "stage main: FAIL"
  overall=1
fi

echo "=== stage recurrentgemma (dev+jax env, +recurrentgemma) $(date +%H:%M:%S) ==="
if ! uv run --isolated --extra dev --extra jax --with recurrentgemma \
    python tpu/pallas_arena/phase0_import_gate.py \
    --stage recurrentgemma --out "$out" --scratch "$scratch"; then
  echo "stage recurrentgemma (pip): FAIL — retrying with clone fallback only"
  if ! uv run --isolated --extra dev --extra jax --with einops --with orbax-checkpoint \
      python tpu/pallas_arena/phase0_import_gate.py \
      --stage recurrentgemma --out "$out" --scratch "$scratch"; then
    echo "stage recurrentgemma: FAIL (both paths)"
    overall=1
  fi
fi

echo "=== stage maxtext (informational, bounded) $(date +%H:%M:%S) ==="
timeout 900 uv run --isolated --extra dev --extra jax --with maxtext \
    python tpu/pallas_arena/phase0_import_gate.py \
    --stage maxtext --out "$out" --scratch "$scratch" \
  || echo "stage maxtext: not importable here (informational only)"

echo "=== phase 0 done, overall=$overall $(date +%H:%M:%S) ==="
rm -rf "$scratch"
exit "$overall"
