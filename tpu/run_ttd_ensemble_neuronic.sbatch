#!/bin/bash
# Ensemble discovery run (two members, shared PUCT pool) on ONE neuronic compute
# node. Grading load = 512 rollouts/step (32 groups x 8 per member x 2 members),
# identical to the single-model 64x8 config. Member spec via TTD_ENSEMBLE_MODELS.
#
# Submit:
#   sbatch --export=ALL,NUM_EPOCHS=15,TTD_ENSEMBLE_MODELS=... \
#     -o <run_dir>/slurm_%j.out tpu/run_ttd_ensemble_neuronic.sbatch
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH -t 2-00:00:00
#SBATCH -J ttd-ensemble
#SBATCH -o /n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_ensemble15/slurm_%j.out
#SBATCH -e /n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_ensemble15/slurm_%j.out
set -euo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
mkdir -p "${TTD_RUN_DIR:-${REPO}/runs/ttd_ensemble15}"

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
