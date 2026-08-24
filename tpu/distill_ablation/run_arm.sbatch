#!/bin/bash
# Run ONE distillation-ablation job script on a single neuronic compute node.
# Grading is in-process across all cores (same as the main runs).
#
# The job to run is a self-contained bash script whose path is in TTD_AB_JOB
# (export it in the launcher, submit with --export=ALL). Generating a concrete
# script file avoids all sbatch quoting / glob-expansion / comma-split hazards.
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
#SBATCH -t 1-00:00:00
#SBATCH -J ttd-ab
set -euo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
VENV="${REPO}/third_party/discover/.venv-ttd-discover"
CORES="${SLURM_CPUS_PER_TASK:-64}"

echo "[ab] node=$(hostname) job=${SLURM_JOB_ID:-none} cores=${CORES}"
echo "[ab] job script: ${TTD_AB_JOB:?set TTD_AB_JOB}"

export TTD_EVAL_BACKEND=local
export TTD_SAFE_GRADE_MAX_WORKERS="${TTD_SAFE_GRADE_MAX_WORKERS:-${CORES}}"
export NUM_CPUS_PER_TASK=1
export TTD_DISCOVER_SYNC=0
export TTD_AB_VENV="${VENV}"
export SSL_CERT_FILE="${SSL_CERT_FILE:-$("${VENV}/bin/python" -c 'import certifi; print(certifi.where())')}"
export REQUESTS_CA_BUNDLE="${REQUESTS_CA_BUNDLE:-${SSL_CERT_FILE}}"
unset TINKER_API_KEY || true

exec bash "${TTD_AB_JOB}"
