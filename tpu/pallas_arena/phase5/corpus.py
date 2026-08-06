"""Phase-5 fleet corpus: ~1600 salted candidates + a duplicate set.

The 20-variant RMSNorm battery is the right *content* (it already spans
instant AST reject -> export-time reject -> full grade, and scores from 0.47
through 1.0 to a deliberately slow kernel) but is far too small to measure
fleet throughput. So each variant is emitted many times with a distinct
SALT COMMENT: semantically the same kernel, a different sha256, therefore a
real independent grade rather than a cache hit. That is what makes the
regrade-spread and cross-judge-agreement numbers meaningful — the same
kernel is genuinely graded ~100 times on 5 different chips.

Two things the multiplicities are NOT free to choose:

  * cost. Phase 4 measured `item_wall_s` per variant; `nondeterministic`
    costs ~10x a normal grade (13.7 s export + 46.2 s compile) and the two
    pathological fixtures cost a whole compile budget each. Emitting those
    at the bulk rate would make the corpus a study of three fixtures. They
    are capped at just enough copies to prove their gate.
  * order. The expensive fixtures are interleaved round-robin, not
    clustered, so no single judge draws the whole slow tail while the other
    four idle — that would make the throughput number a measurement of the
    scheduler rather than of the fleet.

The duplicate set is the SAME text resubmitted after first completion: same
hash, so it must come back from the shared GCS reward cache byte-identical
and near-instantly.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from pallas_arena.phase2.variants import (  # noqa: E402
    WORKER_EXPECT_FAIL_GATE,
    WORKER_EXPECT_PASS,
    WORKER_EXPECT_TERMINAL,
    shakedown_variants,
)

# Relative share of the bulk corpus. Honest variants dominate on purpose:
# they are the ones carrying the reproducibility and cross-judge acceptance
# items, and they are also the realistic mix for an RL rollout stream once
# the AOT pre-gate has removed the non-compiling majority.
BULK_WEIGHTS: dict[str, float] = {
    # honest / must pass
    "honest-xla": 1.0,
    "honest-xla-b": 1.0,
    "honest-xla-f32out": 1.0,
    "unjitted-robust": 1.0,
    "pallas-br16": 1.0,
    "pallas-br256": 1.0,
    "pallas-br512": 1.0,
    "split-personality": 0.6,
    # cheaters that need real fixtures before they die
    "wrong-eps": 0.8,
    "constant-output": 0.8,
    "seed-reader": 0.8,
    "wrong-grad": 0.8,
    # cheap rejects (AST / export / exec)
    "aliased-reference": 0.6,
    "cached-output": 0.5,
    "memoizer": 0.5,
    "obfuscated-import": 0.5,
    "no-kernel": 0.4,
}

# Hard caps for the expensive fixtures, with the phase-4 measured reason.
CAPS: dict[str, int] = {
    "nondeterministic": 8,  # ~60 s/grade (13.7 s export + 46.2 s compile)
    "timer-tamperer": 4,  # deliberately slow kernel; heavy compile
    "compile-bomb": 4,  # each one costs a full compile budget by design
}


def salt(code: str, variant: str, i: int) -> str:
    """Distinct hash, identical semantics. A trailing comment survives
    grader.normalize_code (which only rstrips lines), so the salted copy is
    a cache MISS and gets a real grade."""
    nonce = hashlib.sha256(f"{variant}:{i}".encode()).hexdigest()[:12]
    return f"{code.rstrip()}\n# arena-salt {variant} {i:05d} {nonce}\n"


def plan(target: int = 1600) -> dict[str, int]:
    """variant -> number of salted copies."""
    variants = shakedown_variants()
    counts = {v: n for v, n in CAPS.items() if v in variants}
    remaining = max(target - sum(counts.values()), len(BULK_WEIGHTS))
    wsum = sum(BULK_WEIGHTS.values())
    for v, w in BULK_WEIGHTS.items():
        if v not in variants:
            raise KeyError(f"corpus weight for unknown variant {v!r}")
        counts[v] = max(1, round(remaining * w / wsum))
    return counts


def build(target: int = 1600) -> list[dict]:
    """Round-robin interleaved corpus: [{tag, variant, code}, ...]."""
    variants = shakedown_variants()
    counts = plan(target)
    per_variant = {v: [salt(variants[v], v, i) for i in range(n)] for v, n in counts.items()}
    order = sorted(per_variant, key=lambda v: -counts[v])
    items: list[dict] = []
    idx = {v: 0 for v in order}
    while any(idx[v] < counts[v] for v in order):
        for v in order:
            if idx[v] < counts[v]:
                items.append({"tag": f"{v}#s{idx[v]:05d}", "variant": v, "code": per_variant[v][idx[v]]})
                idx[v] += 1
    return items


def duplicate_set(items: list[dict], n: int = 100) -> list[dict]:
    """A stride-sampled subset resubmitted verbatim: same hash -> the shared
    reward cache must serve a byte-identical verdict. Sampled across the
    whole corpus (not the head) so hits land on entries written by several
    different judges — which is the property a per-judge local cache could
    not deliver."""
    passing = [it for it in items if it["variant"] in WORKER_EXPECT_PASS]
    src = passing or items
    stride = max(len(src) // n, 1)
    picked = src[::stride][:n]
    return [{"tag": it["tag"] + "#dup", "variant": it["variant"], "code": it["code"], "dup_of": it["tag"]} for it in picked]


def expectation(variant: str) -> dict:
    return {
        "expect_pass": variant in WORKER_EXPECT_PASS,
        "gates": WORKER_EXPECT_FAIL_GATE.get(variant),
        "terminal": variant in WORKER_EXPECT_TERMINAL,
    }


if __name__ == "__main__":  # `python -m pallas_arena.phase5.corpus` -> the plan
    import argparse
    import json

    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=1600)
    ap.add_argument("--dups", type=int, default=100)
    a = ap.parse_args()
    items = build(a.target)
    dups = duplicate_set(items, a.dups)
    print(json.dumps({"counts": plan(a.target), "total": len(items), "duplicates": len(dups)}, indent=1))
