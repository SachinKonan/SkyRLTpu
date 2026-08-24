"""Pick ONE fixed set of gpt-oss 'worse' programs (c5 band) to share across all
arms — the fairness anchor. Free/offline. Saves ids + codes so every arm
critiques the SAME programs with matched gaps.
"""

from __future__ import annotations

import argparse
import random

import numpy as np

import common


def vc5(con):
    h = np.asarray(con, float); n = len(h)
    if n == 0 or h.sum() == 0: return None
    h = h * ((n / 2) / h.sum())
    return float(np.max(np.correlate(h, 1 - h, mode="full") * (2.0 / n)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--c5-lo", type=float, default=0.3814)
    ap.add_argument("--c5-hi", type=float, default=0.3818)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--heldout", default="tpu/distill_ablation/heldout_seeds.json")
    ap.add_argument("--out", default="tpu/distill_ablation/corpora/worse_set.json")
    args = ap.parse_args()

    import glob, json
    ho = set(common.read_json(args.heldout)["ids"])
    seen = {}
    for f in glob.glob(args.snapshot):
        for s in json.load(open(f))["states"]:
            if (s.get("code", "").strip() and s.get("construction")
                    and s["id"] not in ho and s["id"] not in seen):
                c5 = vc5(s["construction"])
                if c5 is not None and args.c5_lo <= c5 < args.c5_hi:
                    s["_c5"] = c5
                    seen[s["id"]] = s
    pool = list(seen.values())
    random.Random(args.seed).shuffle(pool)
    chosen = pool[: args.n]
    c5s = sorted(s["_c5"] for s in chosen)
    common.write_json(args.out, {
        "n": len(chosen), "c5_band": [args.c5_lo, args.c5_hi],
        "ids": [s["id"] for s in chosen],
        "worse": [{k: v for k, v in s.items() if not k.startswith("_")} for s in chosen],
    })
    print(f"worse set: {len(chosen)} gpt-oss programs, c5 {c5s[0]:.5f}..{c5s[-1]:.5f} -> {args.out}")


if __name__ == "__main__":
    main()
