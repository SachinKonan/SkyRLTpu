"""production regrade: re-score each cell's top-K fast-ranked programs at FULL budget.

The campaign steered (and reported) with grade_core.grade(fast=True). The gates show that is
sound for SELECTION -- zero top-k regret -- but badly calibrated in MAGNITUDE, and the bias is
not uniform across arms, so cross-arm and cross-baseline comparisons need full-budget numbers.

Emits, per cell: best full score, the fast score it was reported with, and the shift.
"""
import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/rq2/fleet")
import grade_core  # noqa: E402

CELLS = "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2/cells"


def cell_candidates(cell_dir, k, maximize):
    """Top-k distinct programs by the fast score the loop steered on."""
    nodes = []
    for f in ("graph.json", "graph_0.json", "graph_1.json"):
        p = os.path.join(cell_dir, "state", f)
        if not os.path.exists(p):
            continue
        try:
            g = json.load(open(p))
        except Exception:
            continue
        nodes += [x for x in g.get("nodes", [])
                  if x.get("valid") and x.get("r") is not None and x.get("program")]
    if not nodes:
        # workspace arms keep no tree -- state/best.json holds only the single best program,
        # so those cells can only ever be regraded top-1 (noted in the writeup).
        bp = os.path.join(cell_dir, "state", "best.json")
        if os.path.exists(bp):
            try:
                b = json.load(open(bp))
                if b.get("program"):
                    return [{"program": b["program"], "r": b.get("score"), "id": -1}]
            except Exception:
                pass
        return []
    seen, uniq = set(), []
    for x in sorted(nodes, key=lambda x: x["r"], reverse=maximize):
        h = hash(x["program"])
        if h in seen:
            continue
        seen.add(h)
        uniq.append(x)
        if len(uniq) >= k:
            break
    return uniq


def _one(args):
    problem, program, fast, cell, nid = args
    t0 = time.time()
    g = grade_core.grade(problem, program, fast=False, logdir=f"/tmp/regrade_{problem}")
    return {"cell": cell, "id": nid, "fast": fast, "full": g.get("score"),
            "valid": g.get("valid"), "secs": round(time.time() - t0, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=["fc46", "ac1", "erdos"])
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--workers", type=int, default=26)
    ap.add_argument("--only", default=None,
                    help="substring filter on cell name, for patching cells whose candidates all "
                         "timed out on an earlier pass")
    ap.add_argument("--out-suffix", default="",
                    help="append to the output filename so a patch run does not clobber the "
                         "full pass it is patching")
    ap.add_argument("--full-wall", type=int, default=0,
                    help="override grade_core.FULL_WALL; ac1 needs >1100 (24/120 gate programs "
                         "timed out at the default, which silently drops the slowest solvers)")
    a = ap.parse_args()
    if a.full_wall:
        grade_core.FULL_WALL = a.full_wall
    maximize = a.problem == "fc46"
    pick = max if maximize else min

    jobs, cells = [], []
    for d in sorted(os.listdir(CELLS)):
        if not d.startswith(a.problem + "_"):
            continue
        if a.only and a.only not in d:
            continue
        cd = os.path.join(CELLS, d)
        cand = cell_candidates(cd, a.topk, maximize)
        if not cand:
            continue
        cells.append(d)
        jobs += [(a.problem, x["program"], x["r"], d, x["id"]) for x in cand]
    print(f"[regrade] {a.problem}: {len(jobs)} programs over {len(cells)} cells "
          f"(top-{a.topk} each), {a.workers} workers", flush=True)

    rows, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = [ex.submit(_one, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                rows.append({"full": None, "valid": False, "err": str(e)[:100]})
            if n % 25 == 0:
                print(f"[regrade] {n}/{len(jobs)}  {(time.time()-t0)/60:.1f} min", flush=True)

    per = {}
    for r in rows:
        if r.get("full") is None:
            continue
        c = r["cell"]
        if c not in per or (r["full"] > per[c]["full"] if maximize else r["full"] < per[c]["full"]):
            per[c] = r
    out = f"/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2/regrade_{a.problem}{a.out_suffix}.json"
    fast_best = {}
    for d in cells:
        try:
            fast_best[d] = json.load(open(os.path.join(CELLS, d, "result.json")))["best_fast_score"]
        except Exception:
            fast_best[d] = None
    json.dump({"problem": a.problem, "topk": a.topk,
               "per_cell": {c: {"full": v["full"], "fast_reported": fast_best.get(c)}
                            for c, v in per.items()},
               "rows": rows}, open(out, "w"), indent=1)
    print(f"\n[regrade] {a.problem}: full-budget best per cell "
          f"(reported fast -> full, shift)")
    for c in sorted(per, key=lambda c: per[c]["full"], reverse=maximize):
        fb = fast_best.get(c)
        sh = f"{per[c]['full']-float(fb):+.5f}" if fb is not None else "n/a"
        print(f"  {per[c]['full']:.6f}  (was {fb}, {sh})  {c}")
    print(f"[regrade] -> {out}")


if __name__ == "__main__":
    main()
