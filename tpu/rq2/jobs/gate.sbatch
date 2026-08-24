#!/bin/bash
#SBATCH --job-name=rq2-gate
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2/logs/%x_%A_%a.log
# fast-vs-full grading gate. CPU only -- no TPUs, no model calls; programs come off disk.
#   array 0 = fc46 (fast == ONE test case; full == all cases)
#   array 1 = ac1  (fast == program's budget_s forced to 10s; full == its own budget, 1100s wall)
set -uo pipefail
cd /n/fs/vision-mix/sk7524/SkyRLTpu
P=(fc46 ac1)
PROB=${P[$SLURM_ARRAY_TASK_ID]}
# ac1's full grade is ~1100s of wall each, so it gets fewer per cell than cheap fc46
PER=$([ "$PROB" = "fc46" ] && echo 15 || echo 8)
# the problem envs live in the ttt-discover tree and need ITS venv (chz et al) -- `uv run
# --isolated` lacks them, which is how the first attempt died with ModuleNotFoundError: chz.
# Same interpreter + env vars the RQ1 grading jobs use (tpu/rq1/jobs/grade.sh).
PY=/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover/.venv-ttd-discover/bin/python
export TTD_EVAL_BACKEND=local TTD_DISCOVER_SYNC=0
echo "[gate] problem=$PROB per_cell=$PER workers=26 start=$(date '+%F %T')"
$PY tpu/rq2/jobs/gate_fast_full.py --problem "$PROB" --per-cell "$PER" --workers 26
echo "[gate] done $(date '+%F %T')"
