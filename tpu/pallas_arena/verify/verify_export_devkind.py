"""Regression check for the CPU-export device-kind contamination.

MEASURED (job 3687041): 17 of megablox_gmm's 96 prompt-ladder candidates died
at gate `aot_export` with

    ValueError: Unsupported TPU device kind: cpu

Every one of them wrote a RANK-1 `BlockSpec` for a per-row expert-id array
(`pl.BlockSpec((bm,), lambda i, j: (i,))`). Pallas's rank-1 block-shape check
asks the *current device* for its sublane/lane counts, and under
`JAX_PLATFORMS=cpu` the device kind is the string `"cpu"`, so it raises before
the check runs. That is a property of the export environment, not of the
kernel: on a real v6e those blocks satisfy the check. GMM is the only one of
the five arena tasks whose natural formulation has a rank-1 operand, which is
why only GMM hit it.

`judge/child_runner.py:_tpu_export_device_shim` pins the device kind to the
judge's own chip while exporting FOR tpu from a non-tpu child. This script
re-exports the exact 17 contaminated candidates through the identical path and
fails if any of them still dies on the device kind.

Usage:
  JAX_PLATFORMS=cpu python -m pallas_arena.verify.verify_export_devkind \\
      --results runs/pallas_arena/ladder-results-3687041.jsonl --out x.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pallas_arena.probe import configs as C  # noqa: E402
from pallas_arena.probe.pregate import pregate_one, probe_signatures  # noqa: E402

TASK = "megablox_gmm"
NEEDLE = "Unsupported TPU device kind"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="runs/pallas_arena/ladder-results-3687041.jsonl")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout-s", type=float, default=300.0)
    args = ap.parse_args()

    hits = []
    for line in Path(args.results).open():
        r = json.loads(line)
        if r.get("task") == TASK and NEEDLE.split(":")[0] in (r.get("observation") or ""):
            hits.append(r)
    print(f"[devkind] {len(hits)} contaminated candidates in {args.results}", flush=True)
    if not hits:
        print("[devkind] nothing to re-measure", flush=True)
        return 0

    sigs = probe_signatures(TASK, C.TASK_CASES[TASK])[0]
    rows = []
    for i, h in enumerate(hits):
        v = pregate_one(TASK, h["code"], sigs, timeout_s=args.timeout_s)
        obs = (v.get("observation") or "").replace("\n", " | ")
        rows.append(
            {
                "i": i,
                "variant": h.get("variant"),
                "was": "device_kind",
                "now_gate": v.get("gate"),
                "now_passed": bool(v.get("passed")),
                "still_devkind": NEEDLE in obs,
                "observation": obs[:400],
            }
        )
        print(f"[{i:2d}] {h.get('variant')} -> gate={v.get('gate')} passed={v.get('passed')}"
              f"\n     {obs[:280]}", flush=True)

    still = sum(1 for r in rows if r["still_devkind"])
    now_ok = sum(1 for r in rows if r["now_passed"])
    print(f"\n[devkind] still failing on the device kind: {still}/{len(rows)}")
    print(f"[devkind] now clear the export gate:        {now_ok}/{len(rows)}")
    if args.out:
        Path(args.out).write_text(json.dumps({"rows": rows, "still": still, "now_export_ok": now_ok}, indent=1))
    return 1 if still else 0


if __name__ == "__main__":
    raise SystemExit(main())
