#!/bin/bash
# CPU CONTROL for the seam+dialect run. No chips, no engines.
#
# Three earlier runs in this arena published confident all-zero grids that were
# caused by the harness, not the model. So nothing goes to silicon until:
#   1. the GMM export-environment fix is MEASURED on the exact 17 candidates it
#      is supposed to rescue (verify_export_devkind);
#   2. every known-good fill for the three target seams still goes
#      model-style-response -> extract_fill -> compose -> CPU AOT export at all
#      three declared shapes, and computes the right numbers under interpret;
#   3. the sd1/sd2 prompts build, end with the output contract, and fit the
#      served 16384 window with room for a 12000-token completion.
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH -t 1:10:00
#SBATCH --job-name=sd-control
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/pallas_arena/sd-control-%j.log
set -uo pipefail
repo=/n/fs/vision-mix/sk7524/SkyRLTpu
cd "$repo"
export PYTHONPATH="${repo}/tpu:${PYTHONPATH:-}"
mkdir -p runs/pallas_arena
UV_ARGS=(--isolated --extra dev --extra jax --with flatbuffers)
rc_all=0

echo "=== 1. prompts build + token budget $(date +%H:%M:%S) ==="
uv run "${UV_ARGS[@]}" python -m pallas_arena.probe.seam_dialect_prompts_check \
  --out "runs/pallas_arena/sd-prompts-${SLURM_JOB_ID}.json" || rc_all=1

echo
echo "=== 2. GMM export device-kind fix, on the 17 contaminated candidates $(date +%H:%M:%S) ==="
JAX_PLATFORMS=cpu uv run "${UV_ARGS[@]}" python -m pallas_arena.verify.verify_export_devkind \
  --out "runs/pallas_arena/sd-devkind-${SLURM_JOB_ID}.json" || rc_all=1

echo
echo "=== 3. known-good fills through the identical path $(date +%H:%M:%S) ==="
JAX_PLATFORMS=cpu uv run "${UV_ARGS[@]}" python -m pallas_arena.probe.seam_control \
  --tasks splash_attention,ragged_paged_attention,megablox_gmm,rg_lru --interpret \
  --out "runs/pallas_arena/sd-seamcontrol-${SLURM_JOB_ID}.json" || rc_all=1

echo
echo "=== sd control done rc=${rc_all} $(date +%H:%M:%S) ==="
exit "$rc_all"
