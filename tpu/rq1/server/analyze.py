"""RQ1 cross-cell analysis: best-of-N + bootstrap expected-best at matched budgets.

  python3 analyze.py --results runs/fc46_B/result.json runs/fc46_C/result.json ... \
      [--mix C=runs/fc46_C/result.json,D=runs/fc46_D/result.json]

Per cell: valid rate, best-of-all, bootstrap (1000 resamples, fixed RNG) E[best-of-k] for
k in {50, 100}. --mix builds the analysis-only C/D cell by resampling 50/50 from two cells.
Stdlib only.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

KS = (50, 100)
B = 1000


def cell_scores(path):
    r = json.loads(Path(path).read_text())
    # one score PER SUBMISSION (not per unique program): a duplicated program is a real
    # property of the sampling distribution and must count as many draws.
    per_sub = []
    for h, rec in r["results"].items():
        s = rec.get("full", {}).get("score") if rec.get("full", {}).get("valid") else None
        per_sub += [s] * len(rec.get("sessions", [h]))
    return r, per_sub


def boot_best(scores, k, maximize, rng):
    if not any(s is not None for s in scores):
        return None
    acc = 0.0
    for _ in range(B):
        pick = [s for s in rng.choices(scores, k=k) if s is not None]
        if not pick:
            continue
        acc += max(pick) if maximize else min(pick)
    return acc / B


def summarize(name, r, scores, rng):
    maximize = r["maximize"]
    valid = [s for s in scores if s is not None]
    row = {"cell": name, "problem": r["problem"], "n_draws": len(scores),
           "valid_rate": round(len(valid) / max(1, len(scores)), 3),
           "seed_score": r.get("seed_score"),
           "best": (max(valid) if maximize else min(valid)) if valid else None}
    for k in KS:
        row[f"E_best_of_{k}"] = boot_best(scores, k, maximize, rng)
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True)
    ap.add_argument("--mix", default=None,
                    help="NAME=path1,path2 -> 50/50 resampled mixed cell")
    args = ap.parse_args()
    rng = random.Random(31337)
    rows = []
    for p in args.results:
        r, scores = cell_scores(p)
        rows.append(summarize(r.get("cell") or Path(p).parent.name, r, scores, rng))
    if args.mix:
        name, paths = args.mix.split("=")
        p1, p2 = paths.split(",")
        r1, s1 = cell_scores(p1)
        r2, s2 = cell_scores(p2)
        # mixed draw = half from each; bootstrap handles the resampling
        mixed = None
        maximize = r1["maximize"]
        acc = {k: 0.0 for k in KS}
        for _ in range(B):
            for k in KS:
                pick = [s for s in rng.choices(s1, k=k // 2) + rng.choices(s2, k=k - k // 2)
                        if s is not None]
                if pick:
                    acc[k] += max(pick) if maximize else min(pick)
        row = {"cell": name, "problem": r1["problem"],
               "n_draws": len(s1) + len(s2),
               "valid_rate": None, "seed_score": r1.get("seed_score"), "best": None}
        for k in KS:
            row[f"E_best_of_{k}"] = acc[k] / B
        rows.append(row)

    cols = ["cell", "n_draws", "valid_rate", "best"] + [f"E_best_of_{k}" for k in KS]
    seed = rows[0].get("seed_score")
    print(f"problem={rows[0]['problem']}  seed_score={seed}  "
          f"({'maximize' if json.loads(Path(args.results[0]).read_text())['maximize'] else 'minimize'})")
    print(" | ".join(f"{c:>14}" for c in cols))
    for row in rows:
        print(" | ".join(
            f"{row.get(c):>14.6g}" if isinstance(row.get(c), float) else f"{str(row.get(c)):>14}"
            for c in cols))


if __name__ == "__main__":
    main()
