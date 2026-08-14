#!/bin/bash
# Grade the cached Muse-Glimmer rollouts on CPU, sharded across a slurm array.
#
# Usage:
#   sbatch --export=ALL,PROB=erdos,NSH=64 --array=0-63%64 tpu/muse_glimmer/rs_grade.sbatch
#
# 64 shards, not 8. Observed ~346 s/item (worst 992 s), so 960 items serially
# is ~92 h; the previous rounds used 8-16 shards and that is why grading
# crawled. eval_timeout stays 1100 s, the value the Qwen3.5 sweep used.
#
#SBATCH --job-name=mg-rsgrade
#SBATCH --output=/n/fs/vision-mix/sk7524/slurm-logs/mg-rsgrade-%A_%a.log
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -uo pipefail
REPO=/n/fs/vision-mix/sk7524/SkyRLTpu
# .venv-ttd-discover is the one with scipy + cvxpy, which the erdos prompt
# explicitly allows and which every generated program imports. The plain
# .venv has numpy only, and grades every rollout 0 with "No module named
# 'scipy'". numba is deliberately absent from BOTH: the prompt does not
# list it, so a program that reaches for it fails here exactly as it did
# in the sweep.
PY="$REPO/third_party/discover/.venv-ttd-discover/bin/python"
# numba is a DECLARED dependency of the discover package (pyproject.toml:
# "numba>=0.60") but is not installed in .venv-ttd-discover, so round 1 scored
# 67% of its rollouts 0 with `No module named 'numba'` -- an environment
# artifact that was reported as a model behaviour. It is supplied here from an
# isolated --target directory rather than by installing into the shared venv,
# which live rq2 jobs also use. The STRICT reading of the prompt's library
# list (scipy/numpy/cvxpy/math, no numba) is preserved as a separate column:
# rs_analyze.py flags every program that imports numba.
NUMBA_LIBS=/n/fs/vision-mix/sk7524/muse-rs2/pylibs
WORK=/n/fs/vision-mix/sk7524/muse-rs2
PROB="${PROB:-erdos}"
NSH="${NSH:-${SLURM_ARRAY_TASK_COUNT:-64}}"
SH="${SLURM_ARRAY_TASK_ID:-0}"
mkdir -p "$WORK/grades" "$WORK/gradelog/$PROB-$SH"

echo "=== node $(hostname) prob=$PROB shard=$SH/$NSH $(date -Is) ==="
cd "$REPO/third_party/discover" || exit 1
TTD_EVAL_BACKEND=local PYTHONPATH="$REPO/third_party/discover:$NUMBA_LIBS" \
  "$PY" "$REPO/tpu/muse_glimmer/rs_grade.py" \
    --manifest "$WORK/manifests/$PROB.json" \
    --gen "$WORK/gen/$PROB.jsonl" \
    --out "$WORK/grades/$PROB.shard$SH.jsonl" \
    --shard "$SH" --num-shards "$NSH" \
    --eval-timeout "${EVAL_TIMEOUT:-1100}" \
    --log-dir "$WORK/gradelog/$PROB-$SH" \
    --deadline-epoch "${GRADE_DEADLINE:-0}"
echo "=== exit $? $(date -Is) ==="
