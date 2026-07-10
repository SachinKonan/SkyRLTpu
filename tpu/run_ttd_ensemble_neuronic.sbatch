#!/bin/bash
# Ensemble discovery run (gpt-oss-20b + Nemotron-3-Nano, shared PUCT pool) on ONE
# neuronic compute node. Grading load = 512 rollouts/step (32 groups x 8 per
# member x 2 members), identical to the single-model 64x8 config.
#
# Submit (from the ENSEMBLE WORKTREE):
#   sbatch --export=ALL,NUM_EPOCHS=15 tpu/run_ttd_ensemble_neuronic.sbatch
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
# Walltime capped under the maintenance reservation on neu[301-332] (2026-07-14 06:00).
#SBATCH -t 2-00:00:00
#SBATCH -J ttd-ensemble
#SBATCH -o /n/fs/vision-mix/sk7524/SkyRLTpu-ensemble/runs/ttd_ensemble15/slurm_%j.out
#SBATCH -e /n/fs/vision-mix/sk7524/SkyRLTpu-ensemble/runs/ttd_ensemble15/slurm_%j.out
set -euo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu-ensemble
mkdir -p "${REPO}/runs/ttd_ensemble15"

CORES="${SLURM_CPUS_PER_TASK:-64}"
echo "[ensemble] node=$(hostname) job=${SLURM_JOB_ID:-none} cores=${CORES}"

export TTD_EVAL_BACKEND=local
export TTD_SAFE_GRADE_MAX_WORKERS="${TTD_SAFE_GRADE_MAX_WORKERS:-${CORES}}"
export NUM_CPUS_PER_TASK="${NUM_CPUS_PER_TASK:-1}"
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-1100}"
export NUM_EPOCHS="${NUM_EPOCHS:-15}"

# venv prebuilt on shared NFS; don't re-sync inside the job.
export TTD_DISCOVER_SYNC="${TTD_DISCOVER_SYNC:-0}"

exec bash "${REPO}/tpu/run_ttd_ensemble.sh"
