#!/bin/bash
# RQ2 cells at n=200. Throttled to 3 concurrent so the fleet is saturated but not
# oversubscribed (3 x 200 ~= 768 in-flight capacity).
#SBATCH --job-name=rq2_n200
#SBATCH --array=0-35%3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=neu301
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq2/logs/%x_%A_%a.log
set -uo pipefail
RQ2=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/tpu/rq2
CELLS="/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/tpu/rq2/jobs/cells_n200.json"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
cfg=$(python3 -c "
import json,sys
print(json.dumps(json.load(open('$CELLS'))[$SLURM_ARRAY_TASK_ID]))")
read -r problem state execution composition <<< $(python3 -c "
import json; c=json.loads('''$cfg'''); print(c['problem'], c['state'], c['execution'], c['composition'])")
name="${problem}_${state}_${execution}_${composition}_n200"
echo "=== cell $name (array task $SLURM_ARRAY_TASK_ID) ==="
cd "$RQ2/client"
uv run loop.py --problem "$problem" --state "$state" --execution "$execution" \
  --composition "$composition" --n 200 --steps 10 \
  --fast-budget $(python3 -c "
import json; print(json.loads('''$cfg''')['fast_budget'])") \
  --grade-concurrency $(python3 -c "
import json; print(json.loads('''$cfg''')['grade_concurrency'])") \
  --out "/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq2/cells/$name"
