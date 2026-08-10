#!/bin/bash
# Does pinning BLAS threads fix the grading blowup? 6 ud programs at concurrency 6.
# Baseline for comparison: ud_D full grading measured a 9,447 s MEDIAN per program at 30-way
# with threads unconstrained.
#SBATCH --job-name=rq2_threadprobe
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq1/logs/%x_%j.log
set -uo pipefail
PY=/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover/.venv-ttd-discover/bin/python
export TTD_EVAL_BACKEND=local TTD_DISCOVER_SYNC=0
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
echo "threads pinned to 1; grading 6 ud programs at concurrency 6"
$PY - <<'PY'
import json, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed
sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/distill_ablation")
sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/tpu/rq2/../rq1/server")
from grading_mcp import _grade
from pathlib import Path
run = Path("/n/fs/vision-mix/sk7524/SkyRLTpu-rq1/runs/rq1/ud_D")
res = json.loads((run/"result.json").read_text())
hs = [h for h,v in res["results"].items() if v.get("full",{}).get("valid")][:6]
MAIN="/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover"
def job(h):
    code=(run/"solutions"/f"{h}.txt").read_text()
    t0=time.time()
    r=_grade(MAIN,"examples.frontier_erdos_ud.env","FrontierErdosUDEnv","65536","python",
             None, f"```python\n{code}\n```", False, "/tmp/ud_threadprobe")
    return h, round(time.time()-t0,1), r.get("score")
t0=time.time()
with ThreadPoolExecutor(max_workers=6) as ex:
    for f in as_completed([ex.submit(job,h) for h in hs]):
        h,s,sc=f.result(); print(f"  {h} {s:8.1f}s score={sc}", flush=True)
print(f"TOTAL WALL {time.time()-t0:.0f}s for {len(hs)} programs (baseline: ~9447s each at 30-way)")
PY
