"""Re-grade an already-run arm's programs at the production budget, with a full-enabled grader.

Needed because the check-only arm's server ran with --no-full, which also blocked the DRIVER's
yardstick call. The agents' programs are on disk (solutions/<hash>.txt + grade_log sessions), so
this recovers the arm's real numbers with NO codex spend.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent))
import benchmark_cdc as B
from ab_scheme import extract_program, grade_many, start_grader, stop

ap = argparse.ArgumentParser()
ap.add_argument("--run-dir", required=True, help="runs/ab_scheme/<problem><tag>")
ap.add_argument("--arm", required=True)
ap.add_argument("--problem", default="erdos")
ap.add_argument("--group", type=int, default=10)
ap.add_argument("--max-concurrent", type=int, default=12)
args = ap.parse_args()

root = Path(args.run_dir)
adir = root / args.arm
base_json = root / "base_construction.json"
mx = B.MAXIMIZE[args.problem]
prev = json.loads((root / "results.json").read_text()) if (root / "results.json").exists() else {}
seed_prod = prev.get("seed_production")

progs = {}
for g in range(args.group):
    prog, src = extract_program(adir / f"x{g}", adir, f"{args.arm}_x{g}", mx)
    progs[g] = (prog, src)
found = [g for g, (p, _) in progs.items() if p is not None]
print(f"[rg] arm={args.arm} programs recovered: {len(found)}/{args.group} -> {found}", flush=True)

grader, port = start_grader(args.problem, adir, base_json, True, args.max_concurrent)
try:
    jobs = []
    for g, (p, _) in progs.items():
        if p is not None:
            jobs.append((("prod", g), p, "grade_full"))
            jobs.append((("s10", g), p, "check"))
    print(f"[rg] grading {len(jobs)//2} programs concurrently at production 1000s/1100s...",
          flush=True)
    t0 = time.time()
    graded = grade_many(port, jobs) if jobs else {}
    print(f"[rg] wave done in {int(time.time()-t0)}s", flush=True)
finally:
    stop(grader)

rows = []
for g, (p, src) in progs.items():
    if p is None:
        rows.append({"x": g, "prod": None, "s10": None, "note": src})
        print(f"[rg] x{g}: NO PROGRAM ({src})", flush=True)
        continue
    pr, s1 = graded.get(("prod", g), {}), graded.get(("s10", g), {})
    row = {"x": g, "chars": len(p), "src": src,
           "prod": pr.get("score") if pr.get("valid") else None,
           "s10": s1.get("score") if s1.get("valid") else None,
           "note": pr.get("detail", "")[:120]}
    rows.append(row)
    adapt = ("n/a" if row["prod"] is None or row["s10"] is None
             else f"{row['s10'] - row['prod']:+.2e}")
    print(f"[rg] x{g}: production={row['prod']} (10s={row['s10']}, uses-budget={adapt}) [{src}]",
          flush=True)

ok = [r for r in rows if r["prod"] is not None]
best = (max if mx else min)([r["prod"] for r in ok]) if ok else None
beat = ([r for r in ok if (r["prod"] > seed_prod if mx else r["prod"] < seed_prod)]
        if seed_prod is not None else [])
summary = {"rows": rows, "valid": len(ok), "n": args.group, "best": best,
           "beat_baseline": len(beat),
           "mean": (sum(r["prod"] for r in ok) / len(ok)) if ok else None}
out = root / f"regrade_{args.arm}.json"
out.write_text(json.dumps({"baseline": seed_prod, **summary}, indent=2))
print(f"\n[rg] ARM {args.arm} (regraded) baseline={seed_prod} valid={len(ok)}/{args.group} "
      f"beat_baseline={len(beat)}/{args.group} best={best} mean={summary['mean']}", flush=True)
print(f"[rg] wrote {out}")
