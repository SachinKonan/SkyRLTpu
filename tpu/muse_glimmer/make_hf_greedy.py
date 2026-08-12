#!/usr/bin/env python3
"""Extract the small ids-only reference the TPU client needs from hf_ref.npz.

The full dump carries per-layer hidden states and float32 logit rows (hundreds
of MB).  The TPU comparison only needs, per prompt, the exact prompt token ids
and the HF greedy continuation ids -- a few KB, cheap to ship to the VM.
"""
import argparse
import json
import sys

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    z = np.load(args.npz, allow_pickle=False)
    names = sorted({k.split("/")[0] for k in z.files})
    out = {}
    for name in names:
        if f"{name}/greedy_ids" not in z.files:
            continue
        pids = z[f"{name}/ids"].tolist()
        gids = z[f"{name}/greedy_ids"].tolist()
        out[name] = {
            "prompt_ids": pids,
            "greedy_ids": gids,
            "n_prompt_tokens": len(pids),
            "n_greedy_tokens": len(gids),
            "hf_argmax_teacher_forced": z[f"{name}/argmax"].tolist()[-1:],
        }
        print(f"{name:20s} prompt={len(pids):5d} greedy={len(gids):3d}")
    with open(args.out, "w") as fh:
        json.dump(out, fh)
    print(f"wrote {args.out} ({len(out)} prompts)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
