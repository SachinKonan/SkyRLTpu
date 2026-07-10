#!/bin/bash
# Run the full ttt-discover gpt-oss-20b job on ONE neuronic compute node.
#
# neuronic compute nodes reach the internet (verified: api.wandb.ai -> 404,
# tinker.thinkingmachines.dev -> 302), so the entire discover client runs here:
# sampling/training hit the hosted Tinker API directly, and program grading runs
# IN-PROCESS across all allocated cores (TTD_EVAL_BACKEND=local). No Ray, no
# submitit, no dispatch. This is the "grading runs wide" answer for neuronic:
# ~N-wide grading where N = --cpus-per-task (della was capped at 2).
#
# Override knobs (env or sbatch --export):
#   NUM_EPOCHS (default 50), TTD_GRADE_CORES (default 64), EVAL_TIMEOUT (1100),
#   EXPERIMENT_NAME, TTD_RUN_DIR.
#
# Submit:
#   sbatch --export=ALL,NUM_EPOCHS=3 tpu/run_ttd_gptoss20b_neuronic.sbatch   # validation
#   sbatch --export=ALL,NUM_EPOCHS=50 tpu/run_ttd_gptoss20b_neuronic.sbatch  # full run
#
#SBATCH -p all
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G
# Walltime capped under the recurring maintenance reservation on neu[301-332]
# (starts 2026-07-14 06:00). 2 days is ample: 50 epochs finish well under a day.
#SBATCH -t 2-00:00:00
#SBATCH -J ttd-gptoss20b
#SBATCH -o /n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_gptoss20b_full/slurm_%j.out
#SBATCH -e /n/fs/vision-mix/sk7524/SkyRLTpu/runs/ttd_gptoss20b_full/slurm_%j.out
set -euo pipefail

REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
mkdir -p "${REPO}/runs/ttd_gptoss20b_full"

# Number of grading workers = allocated cores on this node. Each grade is a
# 1-core subprocess (NUM_CPUS_PER_TASK=1), so match workers to cores.
CORES="${SLURM_CPUS_PER_TASK:-${TTD_GRADE_CORES:-64}}"

echo "[neuronic] node=$(hostname) job=${SLURM_JOB_ID:-none} cores=${CORES}"
echo "[neuronic] epochs=${NUM_EPOCHS:-50} eval_backend=local grade_workers=${CORES}"

export TTD_EVAL_BACKEND=local
export TTD_SAFE_GRADE_MAX_WORKERS="${TTD_SAFE_GRADE_MAX_WORKERS:-${CORES}}"
export NUM_CPUS_PER_TASK="${NUM_CPUS_PER_TASK:-1}"
# 1100s: the prompt tells generated programs they have budget_s=1000, and they
# time-budget themselves to ~900s. A 300s cap killed ~40% of all rollouts.
export EVAL_TIMEOUT="${EVAL_TIMEOUT:-1100}"
export NUM_EPOCHS="${NUM_EPOCHS:-50}"

# venv is already built on shared NFS; don't re-sync inside the job.
export TTD_DISCOVER_SYNC="${TTD_DISCOVER_SYNC:-0}"

exec bash "${REPO}/tpu/run_ttd_gptoss20b.sh"
