#!/usr/bin/env python3
"""Build the extra references the follow-up TPU run needs, from hf_ref.npz.

Nothing here re-runs HF.  `hf_ref.npz` already carries, for `p4_long_sliding`
(3609 tokens, 1.76x the 2048 sliding window):

  * `ids`      - the exact prompt token ids
  * `argmax`   - HF's teacher-forced argmax at EVERY position
  * `top8_*`   - the top-8 ids/logits at every position
  * `full_rows`- full float32 logit rows at a stride, always including T-1

Because attention is causal, HF's argmax at position i **is** the greedy
next token for the prefix `ids[:i+1]`.  So a one-token greedy request from a
prefix of length L can be checked against `argmax[L-1]` for free, at as many
L as we like.  That is what makes a dense sweep across the 2048 window
boundary cheap: no new HF compute at all.

The top-8 values come along so a divergence can be classified.  A flip whose
top1/top2 gap is ~0 is an argmax tie (see E2E.md section 4, p3_mid), not a KV
bug; a flip with a large gap is a real disagreement.

Also emits the final-position logit rows for the five E2E prompts as a small
npz -- those are the HF distributions the temperature/logprob check compares
against.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# Prefix lengths probed on the TPU.  Dense right across the 2048 window so an
# off-by-one ("first diverges at 2049") is distinguishable from an eviction
# ("first diverges at 3600").
PROBE_LENS = [
    16, 64, 512, 1024, 1536, 2016, 2032, 2040, 2044, 2046, 2047, 2048, 2049,
    2050, 2052, 2056, 2064, 2080, 2100, 2176, 2304, 2560, 3072, 3456, 3584,
    3608, 3609,
]

# Prefix lengths whose multi-token greedy continuation is checked (the decode
# path, as opposed to the prefill-only probes above).  These need a real HF
# generation, produced by hf_ext_ref.py.
DECODE_LENS = [2020, 2040, 2048]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-npz", required=True)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)

    ids = z["p4_long_sliding/ids"].tolist()
    argmax = z["p4_long_sliding/argmax"].tolist()
    top8_ids = z["p4_long_sliding/top8_ids"]
    top8_vals = z["p4_long_sliding/top8_vals"]
    T = len(ids)
    assert len(argmax) == T, (len(argmax), T)

    probes = [L for L in PROBE_LENS if 0 < L <= T]
    out = {
        "source": "hf_ref.npz p4_long_sliding (HF float32, eager)",
        "sliding_window": 2048,
        "p4_ids": ids,
        "p4_n_tokens": T,
        "decode_lens": DECODE_LENS,
        "probes": [],
    }
    for L in probes:
        i = L - 1
        vals = top8_vals[i].astype(np.float64)
        out["probes"].append({
            "prefix_len": L,
            "hf_next_id": int(argmax[i]),
            "hf_top8_ids": top8_ids[i].astype(int).tolist(),
            "hf_top8_vals": vals.tolist(),
            "hf_top1_top2_gap": float(vals[0] - vals[1]),
        })
    Path(args.out_json).write_text(json.dumps(out))
    print(f"wrote {args.out_json}: {len(probes)} probes over p4 ({T} tokens)")
    ties = [p["prefix_len"] for p in out["probes"]
            if p["hf_top1_top2_gap"] < 1e-3]
    print(f"  near-tie probe positions (gap < 1e-3): {ties or 'none'}")

    # Final-position logit rows for the temperature / logprob comparison.
    rows = {}
    for name in sorted({k.split("/")[0] for k in z.files}):
        idx = z[f"{name}/full_rows_idx"].tolist()
        n = len(z[f"{name}/ids"])
        if (n - 1) not in idx:
            print(f"  {name}: no full row at T-1, skipped")
            continue
        rows[f"{name}/final_logits"] = z[f"{name}/full_rows"][idx.index(n - 1)]
        rows[f"{name}/ids"] = z[f"{name}/ids"]
        print(f"  {name}: final logit row {rows[f'{name}/final_logits'].shape}")
    np.savez(args.out_npz, **rows)
    print(f"wrote {args.out_npz}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
