"""GATE for the RQ2 discovery-loop budget: does the FAST check rank candidates the way the
PRODUCTION run does?

The whole RQ2 campaign rests on the loop steering by the fast check (10 s budget) and spending
production grades only on what gets reported -- 13 h of grading instead of 1,660 h. That is only
legitimate if fast scores pick roughly the same winners.

This re-grades programs that ALREADY have production scores on the fast path and reports:
  * Spearman rank correlation over the commonly-valid set
  * validity agreement (does fast reject things production accepts?)
  * SELECTION REGRET -- the decision-relevant number: if the loop keeps the top-k by fast score,
    how much worse is the best production score in that set than the true best?

Run on a compute node with the discover venv:
  $PY fastcheck_gate.py --problem erdos --run-dir <dir with result.json> [--limit N]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path

REPO = "/n/fs/vision-mix/sk7524/SkyRLTpu"
sys.path.insert(0, f"{REPO}/tpu/distill_ablation")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from grading_mcp import _grade  # noqa: E402
from make_problem_pack import PROBLEMS  # noqa: E402

CLIENT_DATA = Path(__file__).resolve().parent.parent / "client" / "data"


def _job(task):
    (root, mod, cls, ptype, lang, constr, payload, logdir) = task
    t0 = time.time()
    r = _grade(root, mod, cls, ptype, lang, constr, payload, True, logdir)   # fast=True
    r["secs"] = round(time.time() - t0, 1)
    return r


def spearman(xs, ys):
    """Rank correlation, average ranks for ties. Stdlib only."""
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        rk = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                rk[order[k]] = avg
            i = j + 1
        return rk
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    dx = sum((a - mx) ** 2 for a in rx) ** 0.5
    dy = sum((b - my) ** 2 for b in ry) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=list(PROBLEMS))
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--concurrency", type=int, default=24)
    args = ap.parse_args()

    root, mod, cls, ptype, lang, maximize = PROBLEMS[args.problem]
    d = CLIENT_DATA / args.problem
    meta = json.loads((d / "meta.json").read_text())
    fence = meta["fence"]
    constr = None
    cj = d / "seed_construction.json"
    if cj.exists():
        constr = json.loads(cj.read_text())

    run = Path(args.run_dir).resolve()
    res = json.loads((run / "result.json").read_text())
    prod = {h: r["full"]["score"] for h, r in res["results"].items()
            if r.get("full", {}).get("valid") and r["full"].get("score") is not None}
    hashes = sorted(prod)[:args.limit] if args.limit else sorted(prod)
    progs = {}
    for h in hashes:
        f = run / "solutions" / f"{h}.txt"
        if f.exists():
            progs[h] = f.read_text()
    print(f"[gate] {args.problem}: {len(progs)} programs with production scores", flush=True)

    logdir = run / "fastcheck_gate"
    logdir.mkdir(exist_ok=True)
    payload = lambda c: f"```{fence}\n{c}\n```" if lang == "python" else c
    tasks = {h: (root, mod, cls, ptype, lang, constr, payload(p), str(logdir))
             for h, p in progs.items()}
    Pool = ProcessPoolExecutor if lang == "cpp" else ThreadPoolExecutor
    fast = {}
    t0 = time.time()
    with Pool(max_workers=args.concurrency) as ex:
        futs = {ex.submit(_job, t): h for h, t in tasks.items()}
        n = 0
        for fut in as_completed(futs):
            h = futs[fut]
            try:
                fast[h] = fut.result()
            except Exception as e:
                fast[h] = {"score": None, "valid": False, "detail": str(e)[:150]}
            n += 1
            if n % 20 == 0:
                print(f"[gate] fast-graded {n}/{len(tasks)}", flush=True)
    secs = [r["secs"] for r in fast.values() if r.get("secs")]
    print(f"[gate] fast pass done in {int(time.time()-t0)}s; "
          f"median {sorted(secs)[len(secs)//2]:.1f}s/program", flush=True)

    both = [h for h in progs if fast.get(h, {}).get("valid") and fast[h].get("score") is not None]
    lost = [h for h in progs if h not in both]
    fs = [fast[h]["score"] for h in both]
    ps = [prod[h] for h in both]
    rho = spearman(fs, ps)

    # selection regret: keep top-k by FAST, how good is the best PRODUCTION score in that set?
    best_true = max(prod.values()) if maximize else min(prod.values())
    rows = []
    for k in (1, 3, 5, 10, 20):
        if k > len(both):
            continue
        pick = sorted(both, key=lambda h: fast[h]["score"], reverse=maximize)[:k]
        best_pick = max(prod[h] for h in pick) if maximize else min(prod[h] for h in pick)
        regret = (best_true - best_pick) if maximize else (best_pick - best_true)
        rel = abs(regret) / abs(best_true) if best_true else float("nan")
        rows.append((k, best_pick, regret, rel))

    out = {"problem": args.problem, "run_dir": str(run), "n_production": len(progs),
           "n_fast_valid": len(both), "n_lost_to_fast": len(lost),
           "spearman": rho, "best_true": best_true,
           "regret": [{"k": k, "best_selected": b, "regret": r, "relative": rl}
                      for k, b, r, rl in rows],
           "median_fast_secs": sorted(secs)[len(secs)//2] if secs else None}
    (run / "fastcheck_gate.json").write_text(json.dumps(out, indent=2))

    print("\n===== FAST-CHECK GATE =====")
    print(f"problem            : {args.problem} ({'maximize' if maximize else 'minimize'})")
    print(f"production programs: {len(progs)}")
    print(f"fast-valid         : {len(both)}  (lost to fast: {len(lost)})")
    print(f"Spearman rho       : {rho:.4f}")
    print(f"true best (prod)   : {best_true:.8g}")
    print("\n  keep top-k by FAST -> best PRODUCTION score in that set")
    for k, b, r, rl in rows:
        print(f"   k={k:2d}: {b:.8g}   regret {r:+.3e}  ({rl:.2%})")
    print("\nread: rho > ~0.6 and small regret at k=10 => steering by fast check is safe")


if __name__ == "__main__":
    main()
