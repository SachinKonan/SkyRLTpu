"""Aggregate the distillation-ablation arms into the comparison tables.

improve = (correctness>0) and (seed_c5 - rollout_c5 > 1e-4)   [make_search_dynamics.py:49]

Usage:
  python analyze.py --runs-dir runs/distill_ablation --arms A0 A1 A2 A3
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

IMPROVE_THRESH = 1e-4
# "taught" dispositions the distilled critiques push (per ANALYSIS.md §3)
SIG_PATTERNS = {
    "finer_grid": re.compile(r"upsample|n_points\s*\*|resolution|finer|refine.*grid|interp", re.I),
    "more_iters": re.compile(r"maxiter|n_iter|iterations|restart|basinhop|anneal", re.I),
}


def load_arm(runs_dir, arm):
    d = Path(runs_dir) / arm
    summ = json.load(open(d / "summary.json")) if (d / "summary.json").exists() else {}
    rollouts = []
    rp = d / "rollouts.jsonl"
    if rp.exists():
        rollouts = [json.loads(l) for l in open(rp)]
    return summ, rollouts


def stats(rollouts):
    """improve-rate, mean Δ (among improvers), taught-signature rate, per stratum + overall."""
    out = {}
    strata = sorted(set(r["stratum"] for r in rollouts)) + ["ALL"]
    for st in strata:
        rs = rollouts if st == "ALL" else [r for r in rollouts if r["stratum"] == st]
        valid = [r for r in rs if r.get("correctness", 0) > 0 and r.get("rollout_c5") is not None]
        n = len(rs)
        improvers = [r for r in valid if (r["seed_c5"] - r["rollout_c5"]) > IMPROVE_THRESH]
        deltas = [r["seed_c5"] - r["rollout_c5"] for r in improvers]
        sig = 0
        for r in improvers:
            code = r.get("parsed_code", "") or ""
            if any(p.search(code) for p in SIG_PATTERNS.values()):
                sig += 1
        out[st] = {
            "n": n, "valid": len(valid),
            "improve_rate": (len(improvers) / len(valid)) if valid else 0.0,
            "mean_delta": (sum(deltas) / len(deltas)) if deltas else 0.0,
            "max_delta": max(deltas) if deltas else 0.0,
            "sig_rate": (sig / len(improvers)) if improvers else 0.0,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", default="runs/distill_ablation")
    ap.add_argument("--arms", nargs="+", required=True)
    args = ap.parse_args()

    print(f"{'arm':<5}{'datums':>7}{'nll0->nllN':>14}{'probe':>8}   "
          f"{'improve% (hard/mid/near/ALL)':>34}{'meanΔ_ALL':>12}{'sig%':>7}")
    rows = {}
    for arm in args.arms:
        summ, rollouts = load_arm(args.runs_dir, arm)
        if not rollouts:
            print(f"{arm:<5} (no rollouts yet)")
            continue
        s = stats(rollouts)
        rows[arm] = (summ, s)
        nll = summ.get("nll_trace") or []
        nll_str = f"{nll[0]:.2f}->{nll[-1]:.2f}" if nll else "—"
        probe = summ.get("probe_nll")
        probe_str = f"{probe:.3f}" if probe is not None else "—"
        imp = "/".join(f"{s[k]['improve_rate']*100:.0f}" for k in ["hard", "mid", "near", "ALL"] if k in s)
        print(f"{arm:<5}{summ.get('n_datums',0):>7}{nll_str:>14}{probe_str:>8}   "
              f"{imp:>34}{s['ALL']['mean_delta']:>12.2e}{s['ALL']['sig_rate']*100:>6.0f}%")

    # comparisons
    if "A0" in rows:
        base = rows["A0"][1]["ALL"]["improve_rate"]
        print(f"\nbaseline (A0) ALL improve-rate: {base*100:.1f}%")
        for arm in args.arms:
            if arm != "A0" and arm in rows:
                r = rows[arm][1]["ALL"]["improve_rate"]
                print(f"  {arm}: {r*100:.1f}%  (Δ vs A0 = {(r-base)*100:+.1f} pts)")


if __name__ == "__main__":
    main()
