"""Extract a fixed, difficulty-stratified held-out seed battery from a rich pool
snapshot. These seeds are the improver-eval battery; they are EXCLUDED (by id)
from every training corpus. Free/offline — no Tinker.

Usage:
  python extract_heldout.py \
    --snapshot .../ttd_gptoss20b_ctrl15/tinker_log/*/puct_sampler_step_000015.json \
    --per-stratum 6 --out tpu/distill_ablation/heldout_seeds.json
"""

from __future__ import annotations

import argparse
import glob

import numpy as np

import common


def verify_c5(construction):
    h = np.asarray(construction, dtype=np.float64)
    n = len(h)
    t = n / 2.0
    h2 = h * (t / h.sum()) if h.sum() != t else h
    return float(np.max(np.correlate(h2, 1.0 - h2, mode="full") * (2.0 / n))), n


# difficulty by achieved c5 (higher c5 = coarser/further = "easier to improve").
# Bands set from the union-of-snapshots c5 distribution for ctrl15.
STRATA = [
    ("hard",  lambda c5: 0.3820 <= c5 < 0.3860),  # coarse real programs, lots of headroom
    ("mid",   lambda c5: 0.3813 <= c5 < 0.3820),
    ("near",  lambda c5: 0.3808 <= c5 < 0.3813),   # near-frontier, hard to improve
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True,
                    help="puct_sampler_step_*.json glob; UNIONed across all matches")
    ap.add_argument("--per-stratum", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="tpu/distill_ablation/heldout_seeds.json")
    args = ap.parse_args()

    snaps = sorted(glob.glob(args.snapshot))
    if not snaps:
        raise SystemExit(f"no snapshots match {args.snapshot}")
    rng = np.random.default_rng(args.seed)

    # union coded states across all snapshots, dedup by id (early steps hold the
    # coarse programs later pruned from the converged pool)
    seen: dict[str, dict] = {}
    for snap in snaps:
        for s in common.read_json(snap)["states"]:
            if s.get("code", "").strip() and s.get("construction") and s.get("value") is not None:
                seen.setdefault(s["id"], s)
    coded = list(seen.values())
    for s in coded:
        s["_c5"], s["_n"] = verify_c5(s["construction"])
    snap = f"{snaps[0]} .. {snaps[-1]} (union, {len(snaps)} snapshots)"

    chosen, chosen_ids = [], set()
    for name, pred in STRATA:
        cands = [s for s in coded if pred(s["_c5"]) and s["id"] not in chosen_ids]
        if not cands:
            print(f"[warn] stratum {name}: no candidates")
            continue
        # spread across n: sort by n, take evenly spaced picks
        cands.sort(key=lambda s: s["_n"])
        k = min(args.per_stratum, len(cands))
        idx = np.linspace(0, len(cands) - 1, k).round().astype(int)
        # small jitter so re-runs with the same seed are stable but not degenerate
        for i in idx:
            s = cands[int(i)]
            if s["id"] in chosen_ids:
                continue
            chosen.append({"stratum": name, "c5": s["_c5"], "n": s["_n"], "state": {
                k2: v for k2, v in s.items() if not k2.startswith("_")
            }})
            chosen_ids.add(s["id"])

    out = {
        "source_snapshot": snap,
        "count": len(chosen),
        "ids": sorted(chosen_ids),
        "seeds": chosen,
    }
    common.write_json(args.out, out)

    print(f"Held-out battery: {len(chosen)} seeds -> {args.out}")
    for name, _ in STRATA:
        grp = [c for c in chosen if c["stratum"] == name]
        if grp:
            c5s = [c["c5"] for c in grp]
            ns = [c["n"] for c in grp]
            print(f"  {name:<5} {len(grp)} seeds  c5 {min(c5s):.6f}..{max(c5s):.6f}  n {min(ns)}..{max(ns)}")


if __name__ == "__main__":
    main()
