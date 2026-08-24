"""Stage-1 decoupled-experiment table: per-executor (from regrade_*.json, authoritative,
graded off the cached workdirs) vs the direct-vanilla baseline (in_context_improve.py,
gpt-oss writes code itself). Answers: (1) does delegating execution to Codex raise
VALIDITY above gpt-oss's ~44%? (2) does it raise bases-improved / best c5?
"""

from __future__ import annotations

import json
import os
import statistics

AB = os.path.dirname(os.path.abspath(__file__))


def load(path):
    p = os.path.join(AB, path)
    return json.load(open(p)) if os.path.exists(p) else None


def exec_row(reg):
    res = reg["results"]
    n = len(res)
    c5s = sorted(r["c5"] for r in res if r["c5"] is not None)
    per_base = {}
    for r in res:
        per_base.setdefault(r["base_id"], []).append((r["base_c5"], r["c5"]))
    bimp = sum(1 for v in per_base.values()
               if any(c is not None and bc - c > 1e-4 for bc, c in v))
    return {"label": reg["executor"], "got": n, "valid": len(c5s),
            "vpct": 100 * len(c5s) / max(n, 1), "bases": len(per_base), "bimp": bimp,
            "best": c5s[0] if c5s else None, "med": c5s[len(c5s) // 2] if c5s else None}


def direct_row(d):
    res = d["results"]
    attempts = sum(len(r["results"]) for r in res)
    c5s = sorted(c for r in res for c in r["results"] if c is not None)
    bimp = sum(r["improved"] for r in res)
    return {"label": "DIRECT (gpt-oss writes code)", "got": attempts, "valid": len(c5s),
            "vpct": 100 * len(c5s) / max(attempts, 1), "bases": len(res), "bimp": bimp,
            "best": c5s[0] if c5s else None, "med": c5s[len(c5s) // 2] if c5s else None}


def fmt(x):
    return f"{x:.6f}" if isinstance(x, float) else str(x)


def main():
    rows = []
    for tag, lbl in [("corpora/regrade_van_mini.json", "mini"),
                     ("corpora/regrade_van_terra.json", "terra"),
                     ("corpora/regrade_van_luna.json", "luna")]:
        reg = load(tag)
        if reg:
            rows.append(exec_row(reg))
    direct = load("corpora/direct_vanilla.json")
    if direct:
        rows.append(direct_row(direct))

    print("\n================ STAGE 1: decoupled executor comparison (vanilla context) ================")
    print(f"{'arm':<30} {'got':>5} {'valid':>7} {'valid%':>7} {'bases-imp':>10} {'best c5':>10} {'median':>10}")
    print("-" * 92)
    for r in rows:
        print(f"{r['label']:<30} {r['got']:>5} {r['valid']:>7} {r['vpct']:>6.0f}% "
              f"{r['bimp']:>4}/{r['bases']:<5} {fmt(r['best']):>10} {fmt(r['med']):>10}")

    execs = [r for r in rows if r["label"] != "DIRECT (gpt-oss writes code)"]
    dr = next((r for r in rows if r["label"].startswith("DIRECT")), None)
    if execs and dr:
        best_exec = min(execs, key=lambda r: (r["best"] is None, r["best"]))
        print("\n---- key contrasts ----")
        print(f"  best executor by c5: {best_exec['label']} (best {fmt(best_exec['best'])}, "
              f"valid {best_exec['vpct']:.0f}%, bases-imp {best_exec['bimp']}/{best_exec['bases']})")
        print(f"  VALIDITY: decoupled(best) {best_exec['vpct']:.0f}%  vs  direct {dr['vpct']:.0f}%  "
              f"(Δ {best_exec['vpct']-dr['vpct']:+.0f} pts)")
        db = "n/a" if (best_exec['best'] is None or dr['best'] is None) else f"{dr['best']-best_exec['best']:+.6f}"
        print(f"  BEST c5: decoupled(best) {fmt(best_exec['best'])}  vs  direct {fmt(dr['best'])}  (Δ {db})")
        print(f"  BASES-IMPROVED: decoupled(best) {best_exec['bimp']}/{best_exec['bases']}  "
              f"vs  direct {dr['bimp']}/{dr['bases']}")


if __name__ == "__main__":
    main()
