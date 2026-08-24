"""fast-vs-full grading gate for fc46 and ac1 -- the check the campaign never ran.

The RQ2 loop graded every program with grade_core.grade(fast=True): for python problems that
rewrites the program's own `budget_s=N` down to 10s, and for fc46 (cpp) it evaluates ONE test
case instead of all of them. Only erdos was ever gated (rho=0.907, zero top-20 selection
regret). This script closes the gap for the other two.

Method: sample N programs spanning the fast-score range from the finished cells' PUCT graphs
(state/graph.json carries `program` + `r` = the fast score it was steered by), regrade each at
FULL budget, and report
  * spearman rho between fast and full,
  * top-k selection regret: (best full score among the fast-top-k) vs (best full score overall),
    for k in 1,5,10,20 -- the number that decides how many programs a production regrade must
    cover per cell.

CPU-only, no TPUs, no model calls. Programs come off disk.
"""
import argparse, json, os, random, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.insert(0, "/n/fs/vision-mix/sk7524/SkyRLTpu/tpu/rq2/fleet")
import grade_core  # noqa: E402

CELLS = "/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2/cells"


def collect(problem, per_cell, seed):
    """Stratified sample: split each cell's valid programs into per_cell score bands and take one
    from each, so the sample spans weak->strong rather than clustering at the frontier (a sample
    of only good programs would inflate rho by restricting range)."""
    rng = random.Random(seed)
    picked = []
    for d in sorted(os.listdir(CELLS)):
        if not d.startswith(problem + "_"):
            continue
        # single-buffer arms carry graph.json; team-split keeps one buffer per agent
        graphs = [p for p in (os.path.join(CELLS, d, "state", f)
                              for f in ("graph.json", "graph_0.json", "graph_1.json"))
                  if os.path.exists(p)]
        nodes, maximize = [], None
        for gp in graphs:
            try:
                g = json.load(open(gp))
            except Exception:
                continue
            maximize = bool(g.get("maximize")) if maximize is None else maximize
            nodes += [x for x in g.get("nodes", [])
                      if x.get("valid") and x.get("r") is not None and x.get("program")]
        if not nodes:
            continue
        nodes.sort(key=lambda x: x["r"], reverse=bool(maximize))
        band = max(1, len(nodes) // per_cell)
        for i in range(per_cell):
            chunk = nodes[i * band:(i + 1) * band]
            if chunk:
                x = rng.choice(chunk)
                picked.append({"cell": d, "id": x["id"], "fast": x["r"], "program": x["program"]})
    return picked


def _full(item, problem):
    t0 = time.time()
    g = grade_core.grade(problem, item["program"], fast=False,
                         logdir=f"/tmp/gate_full_{problem}")
    return {**{k: item[k] for k in ("cell", "id", "fast")},
            "full": g.get("score"), "valid": g.get("valid"),
            "detail": (g.get("detail") or "")[:120], "secs": round(time.time() - t0, 1)}


def spearman(xs, ys):
    def ranks(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and v[order[j + 1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = ranks(xs), ranks(ys)
    n = len(xs)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = (sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry)) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problem", required=True, choices=["fc46", "ac1", "erdos"])
    ap.add_argument("--per-cell", type=int, default=10)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    maximize = a.problem == "fc46"
    items = collect(a.problem, a.per_cell, a.seed)
    out = a.out or f"/n/fs/vision-mix/sk7524/SkyRLTpu/runs/rq2/gate_{a.problem}.json"
    print(f"[gate] {a.problem}: {len(items)} programs from "
          f"{len(set(i['cell'] for i in items))} cells, {a.workers} workers", flush=True)

    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(_full, it, a.problem): it for it in items}
        for n, f in enumerate(as_completed(futs), 1):
            try:
                rows.append(f.result())
            except Exception as e:
                rows.append({**{k: futs[f][k] for k in ("cell", "id", "fast")},
                             "full": None, "valid": False, "detail": f"crash: {e}"[:120]})
            if n % 10 == 0:
                print(f"[gate] {n}/{len(items)}  {(time.time()-t0)/60:.1f} min", flush=True)
            json.dump(rows, open(out, "w"), indent=1)

    ok = [r for r in rows if r.get("full") is not None]
    print(f"\n[gate] {a.problem}: {len(ok)}/{len(rows)} full-valid")
    if len(ok) < 8:
        print("[gate] too few valid to report"); return
    rho = spearman([r["fast"] for r in ok], [r["full"] for r in ok])
    pick = max if maximize else min
    best_full = pick(r["full"] for r in ok)
    ok.sort(key=lambda r: r["fast"], reverse=maximize)     # rank by the score the loop steered on
    print(f"[gate] spearman rho(fast, full) = {rho:.4f}   n={len(ok)}")
    print(f"[gate] best full over sample = {best_full}")
    for k in (1, 5, 10, 20):
        if k <= len(ok):
            got = pick(r["full"] for r in ok[:k])
            regret = abs(got - best_full)
            print(f"[gate] top-{k:<2d} regret = {regret:.6g}   (picks {got})")
    json.dump({"problem": a.problem, "n": len(ok), "rho": rho, "rows": rows},
              open(out, "w"), indent=1)
    print(f"[gate] -> {out}")


if __name__ == "__main__":
    main()
