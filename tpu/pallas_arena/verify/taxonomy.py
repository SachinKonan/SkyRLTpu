"""Root-cause taxonomy of every seam-probe failure, from the cached rows.

`aot_export` was 63% of all failures and export is a CPU event, so the whole
diagnosis is CPU work on `runs/pallas_arena/seam-results-*.jsonl`. The gate
histogram in SEAM-REPORT.md says WHERE candidates die; this says WHY, clustered
by the single fact a prompt would have to state to prevent it.

Usage: python -m pallas_arena.verify.taxonomy --results <jsonl> --out <json>
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys

# Each cluster: (id, human cause, the prompt/harness fact that prevents it,
# predicate over (gate, observation, code)).
CAUSES: list[tuple[str, str, str, object]] = [
    (
        "hallucinated-pallas-api",
        "invented a name or kwarg that does not exist at the pin",
        "API-signature block (already present in `seam`; absent in `reference`)",
        lambda g, o, c: bool(
            re.search(
                r"unexpected keyword argument '(out_spec|out_dtype|out_dtypes|out_shapes|in_shapes|"
                r"block_shapes|index_map|scalar_prefetch|out_sharding)'"
                r"|has no attribute '(pallas_call|Ref|InSpec|ANY|TpuCompilerParams|TPUCompilerParams|"
                r"thread_idx|for_i_loop)'"
                r"|cannot import name 'pltpu'"
                r"|has no attribute 'Shape'",
                o,
            )
        ),
    ),
    (
        "dot-general-dimension-numbers",
        "`dimension_numbers` passed FLAT instead of ((lc,),(rc,)),((lb,),(rb,))",
        "one worked `jax.lax.dot_general` call in the API block",
        lambda g, o, c: (
            ("values to unpack" in o and "dot_general" in (c or ""))
            or "dot_general requires" in o
            or "requires lhs batch dimensions" in o
            or "requires contracting dimensions" in o
        ),
    ),
    (
        "ref-store-does-not-broadcast",
        "wrote a SCALAR into a Ref: `m_ref[...] = -jnp.inf`",
        "'a Ref store is not a broadcast: the RHS must already have the Ref's shape'",
        lambda g, o, c: "Invalid shape for `swap`" in o or "Invalid shape for swap" in o,
    ),
    (
        "vmem-spec-used-as-buffer",
        "called `pltpu.VMEM(...)` inside the kernel body and indexed it",
        "'pltpu.VMEM is a SPEC for scratch_shapes=, not an allocation; you cannot "
        "allocate scratch inside a kernel body'",
        lambda g, o, c: "'MemoryRef' object does not support item assignment" in o,
    ),
    (
        "pl-when-called-after-decorating",
        "`@pl.when(...)` then called the decorated function / used it as a ctx mgr",
        "'@pl.when consumes the function and returns None -- never call it, never `with` it'",
        lambda g, o, c: (
            "'NoneType' object is not callable" in o
            or "does not support the context manager protocol" in o
        ),
    ),
    (
        "traced-value-in-python-control-flow",
        "python `if` / `bool()` on a traced array",
        "'a Ref read is a tracer: use jnp.where or @pl.when, never `if`'",
        lambda g, o, c: "TracerBoolConversionError" in o or "ConcretizationTypeError" in o,
    ),
    (
        "associative-scan-combine-arity",
        "`lax.associative_scan` combine fn written with 4 scalars / 1 arg "
        "instead of two PYTREES",
        "the combine signature is already in the rg_lru seam prose -- it needs to be "
        "in the STUB the model copies, not in prose",
        lambda g, o, c: bool(
            re.search(r"(combine|assoc|affine)\w*\(\) (takes 1 positional argument but 2 were given"
                      r"|missing \d+ required positional argument)", o)
            or "associative_scan: fn argument should be callable" in o
        ),
    ),
    (
        "chunk-return-layout",
        "returned the chunk with time first, so the harness's [b, d] carry reshape blew up",
        "state the two return SHAPES as an assert in the scaffold, or have the harness "
        "take h_last itself (h_chunk[:, -1])",
        lambda g, o, c: bool(re.search(r"cannot reshape array of shape \(\d+, \d+, \d+\) .* into shape \(\d+, \d+\)", o)),
    ),
    (
        "returned-a-function",
        "the entrypoint returned a jitted function instead of an array",
        "'return the ARRAY, not the function' in the output-format block",
        lambda g, o, c: "PjitFunction" in o and "not a valid JAX type" in o,
    ),
    (
        "fill-return-contract",
        "the fill function returned the wrong number of values",
        "narrower fill signature / an explicit `return` line in the seam stub",
        lambda g, o, c: "values to unpack" in o,
    ),
    (
        "blockspec-shape-mismatch",
        "BlockSpec rank / Mosaic tiling / index-map arity",
        "harness owns the BlockSpecs (already true under `seam`)",
        lambda g, o, c: bool(
            re.search(r"Block shape for args|divisible by 8 and 128|Index map function|"
                      r"Pytree for `in_specs`", o)
        ),
    ),
    (
        "kernel-body-arity",
        "the pallas kernel body signature did not match the call",
        "harness owns the pallas_call (already true under `seam`)",
        lambda g, o, c: bool(re.search(r"missing \d+ required (keyword-only |positional )?argument", o)),
    ),
    (
        "shape-algebra",
        "in-kernel shapes did not line up (broadcast / axis / rank)",
        "a worked example kernel at the same seam",
        lambda g, o, c: bool(
            re.search(r"Incompatible shapes for broadcasting|axis \d+ is out of bounds|"
                      r"is not iterable|not subscriptable|Non-static stride", o)
        ),
    ),
    (
        "undefined-name",
        "referenced a name it never defined (usually a truncated or garbled body)",
        "shorter fills / a token budget the model respects",
        lambda g, o, c: "NameError" in o,
    ),
    (
        "truncation-syntax",
        "generation ran out of tokens mid-program",
        "smaller fill budget; `reference` variant is 3-10x over budget",
        lambda g, o, c: g == "ast" and ("syntax error" in o or "was never closed" in o or "unterminated" in o),
    ),
    (
        "no-pallas-call",
        "never emitted a pallas_call at all (task requires one)",
        "the seam supplies the pallas_call; `reference` does not",
        lambda g, o, c: "no pallas_call found" in o,
    ),
    (
        "numerics",
        "compiled and ran, but missed the calibrated tolerance",
        "precision knob prose (already present)",
        lambda g, o, c: g == "correctness",
    ),
    (
        "gradient",
        "forward correct, backward wrong",
        "split the task into a fwd-only stage first",
        lambda g, o, c: g == "gradient",
    ),
]


def classify(gate: str, obs: str, code: str) -> list[str]:
    obs = obs or ""
    hits = [cid for cid, _c, _f, pred in CAUSES if pred(gate, obs, code)]
    return hits or ["unclassified"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.results)]
    causes = {cid: (c, f) for cid, c, f, _ in CAUSES}
    per_task = collections.defaultdict(lambda: collections.defaultdict(list))
    tally = collections.Counter()
    examples: dict[tuple, str] = {}

    for r in rows:
        if r["gate"] == "all":
            continue
        obs = r.get("observation") or ""
        hits = classify(r["gate"], obs, r.get("code") or "")
        primary = hits[0]
        key = (r["task"], r["variant"], primary)
        per_task[r["task"]][primary].append(f"{r['model']}|{r['variant']}|i{r['idx']}")
        tally[(r["task"], primary)] += 1
        if key not in examples:
            examples[key] = obs.strip().replace("\n", " | ")[:240]

    report = {"per_task": {}, "totals": {}}
    for task in sorted(per_task):
        print("=" * 96)
        n_fail = sum(len(v) for v in per_task[task].values())
        print(f"{task}   {n_fail} failing candidates")
        rows_out = []
        for cid, who in sorted(per_task[task].items(), key=lambda kv: -len(kv[1])):
            cause, fix = causes.get(cid, (cid, "-"))
            ex = next((examples[k] for k in examples if k[0] == task and k[2] == cid), "")
            print(f"  {len(who):3d}  {cid:34s} {cause}")
            print(f"       FIX: {fix}")
            if ex:
                print(f"       e.g. {ex}")
            rows_out.append({"cause": cid, "n": len(who), "who": who, "human": cause,
                             "fix": fix, "example": ex})
        report["per_task"][task] = rows_out
    for (task, cid), n in tally.most_common():
        report["totals"][f"{task}::{cid}"] = n

    if args.out:
        json.dump(report, open(args.out, "w"), indent=1)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
