"""Enumerate the RQ2 cells and emit one sbatch array per scale.

Scale gets its own array because the throttle must match the fleet's in-flight capacity: 24
workers x 32 concurrent = 768 requests in flight, so ~7 cells can run together at n=100 but only
1-2 at n=500. One array with a single throttle would either oversubscribe the farm at large n or
idle it at small n.

Cells are ordered to INTERLEAVE compositions (qwen / gemma / 50-50 round-robin). A wave of
qwen-only cells would leave the 12 gemma workers idle; interleaving keeps both pools busy, which
is worth real wall-clock at 24 workers.

  python3 make_cells.py --problems fc46 erdos ac1 --out jobs/
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

STATES = ["puct", "workspace"]
EXECUTIONS = ["simple", "orchestrator"]
COMPOSITIONS = ["qwen", "gemma", "50-50"]
# per-problem fast budget: fidelity of the fast ranking is budget-dependent and differs by
# problem. Measured Spearman vs production -- fc46 0.97 @1 case, erdos 0.91 @10s, ac1 0.68 @10s
# but 0.85 @60s, so ac1 buys its fidelity with budget.
FAST_BUDGET = {"fc46": 10, "erdos": 10, "ac1": 60, "fc159": 10, "ud": 10}
# scale = G sweep at FIXED B=16 (chains constant across scales, so chain depth stays
# comparable and "scale" cleanly means more rollouts per sampled state, not a different tree)
B = 16
G_SCALES = [32, 64]             # n = B*G in {512, 1024} -- 16x32 is the RL shape
# concurrent cells per scale, from 768 in-flight capacity
THROTTLE = {512: 2, 1024: 1}
# per-cell client concurrency: throttle * conc ~= fleet in-flight capacity (768). The loop's
# default of 64 would starve a 1024-rollout step at 24 workers.
CLIENT_CONC = {512: 384, 1024: 768}
GRADE_CONC = {"fc46": 8, "fc159": 8, "erdos": 16, "ac1": 16, "ud": 16}

ARRAY = """#!/bin/bash
# RQ2 cells at n={n}. Throttled to {throttle} concurrent so the fleet is saturated but not
# oversubscribed ({throttle} x {n} ~= 768 in-flight capacity).
#SBATCH --job-name=rq2_n{n}
#SBATCH --array=0-{last}%{throttle}
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task={cpus}
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --exclude=neu301
#SBATCH --output={runs}/logs/%x_%A_%a.log
set -uo pipefail
RQ2=/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/rq2
CELLS="{cells_json}"
export PATH="/n/fs/vision-mix/sk7524/.npm-global/bin:$PATH"
PY=/n/fs/vision-mix/sk7524/SkyRLTpu/third_party/discover/.venv-ttd-discover/bin/python
export TTD_EVAL_BACKEND=local TTD_DISCOVER_SYNC=0
cfg=$(python3 -c "
import json,sys
print(json.dumps(json.load(open('$CELLS'))[$SLURM_ARRAY_TASK_ID]))")
read -r problem state execution composition <<< $(python3 -c "
import json; c=json.loads('''$cfg'''); print(c['problem'], c['state'], c['execution'], c['composition'])")
name="${{problem}}_${{state}}_${{execution}}_${{composition}}_n{n}"
echo "=== cell $name (array task $SLURM_ARRAY_TASK_ID) ==="
cd "$RQ2/client"
$PY loop.py --problem "$problem" --state "$state" --execution "$execution" \\
  --composition "$composition" --B {B} --G {G} --steps {steps} \\
  --concurrency {conc} \\
  --fast-budget $(python3 -c "
import json; print(json.loads('''$cfg''')['fast_budget'])") \\
  --grade-concurrency $(python3 -c "
import json; print(json.loads('''$cfg''')['grade_concurrency'])") \\
  --out "{runs}/cells/$name"
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", nargs="+", default=["fc46", "erdos", "ac1"])
    ap.add_argument("--scales", nargs="+", type=int, default=G_SCALES,
                    help="G values; n = 16*G per step")
    ap.add_argument("--steps", type=int, default=10)
    ap.add_argument("--runs", default="/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2")
    ap.add_argument("--out", default=str(Path(__file__).parent))
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for g in args.scales:
        n = B * g
        cells = []
        # interleave compositions so both model pools stay busy
        combos = list(itertools.product(args.problems, STATES, EXECUTIONS))
        for ci, comp in enumerate(COMPOSITIONS):
            for (prob, st, ex) in combos:
                cells.append({"problem": prob, "state": st, "execution": ex,
                              "composition": comp, "n": n, "B": B, "G": g,
                              "fast_budget": FAST_BUDGET.get(prob, 10),
                              "grade_concurrency": GRADE_CONC.get(prob, 16)})
        cells.sort(key=lambda c: COMPOSITIONS.index(c["composition"]))
        inter = []
        by_comp = {c: [x for x in cells if x["composition"] == c] for c in COMPOSITIONS}
        for i in range(max(len(v) for v in by_comp.values())):
            for c in COMPOSITIONS:
                if i < len(by_comp[c]):
                    inter.append(by_comp[c][i])
        cells = inter

        cj = out / f"cells_n{n}.json"
        cj.write_text(json.dumps(cells, indent=2))
        sh = out / f"run_n{n}.sh"
        sh.write_text(ARRAY.format(
            n=n, B=B, G=g, conc=CLIENT_CONC.get(n, 384),
            last=len(cells) - 1, throttle=THROTTLE.get(n, 2),
            cpus=max(8, min(32, GRADE_CONC.get(args.problems[0], 16) + 8)),
            cells_json=cj, runs=args.runs, steps=args.steps))
        sh.chmod(0o755)
        total += len(cells)
        print(f"B={B} G={g:2d} (n={n:3d}): {len(cells):2d} cells, throttle {THROTTLE.get(n,2)} -> {sh}")
    print(f"\ntotal {total} cells across {len(args.scales)} arrays "
          f"({len(args.problems)} problems x {len(STATES)}x{len(EXECUTIONS)} treatments "
          f"x {len(COMPOSITIONS)} compositions)")
    print(f"programs: {sum(len(json.loads((out/f'cells_n{B*g}.json').read_text()))*args.steps*B*g for g in args.scales):,}")


if __name__ == "__main__":
    main()
