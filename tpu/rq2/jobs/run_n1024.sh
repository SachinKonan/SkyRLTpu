#!/bin/bash
# RQ2 cells at n=1024. Throttled to 1 concurrent so the fleet is saturated but not
# oversubscribed (1 x 1024 ~= 768 in-flight capacity).
#SBATCH --job-name=rq2_n1024
#SBATCH --array=0-35%1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=neu301
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2/logs/%x_%A_%a.log
set -uo pipefail
RQ2=/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/rq2
CELLS="/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/rq2/jobs/cells_n1024.json"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
PY=/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover/.venv-ttd-discover/bin/python
export TTD_EVAL_BACKEND=local TTD_DISCOVER_SYNC=0
cfg=$(python3 -c "
import json,sys
print(json.dumps(json.load(open('$CELLS'))[$SLURM_ARRAY_TASK_ID]))")
read -r problem state execution composition <<< $(python3 -c "
import json; c=json.loads('''$cfg'''); print(c['problem'], c['state'], c['execution'], c['composition'])")
name="${problem}_${state}_${execution}_${composition}_n1024"
echo "=== cell $name (array task $SLURM_ARRAY_TASK_ID) ==="
cd "$RQ2/client"
$PY loop.py --problem "$problem" --state "$state" --execution "$execution" \
  --composition "$composition" --B 16 --G 64 --steps 10 \
  --concurrency 768 \
  --fast-budget $(python3 -c "
import json; print(json.loads('''$cfg''')['fast_budget'])") \
  --grade-concurrency $(python3 -c "
import json; print(json.loads('''$cfg''')['grade_concurrency'])") \
  --out "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2/cells/$name"
