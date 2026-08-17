"""Bisect the v6e boot SIGSEGV (job 3714344) to a single compiled graph.

The crash: a STACK OVERFLOW inside the TPU compiler's convolution cost model
(`FusedSpatialMajorConvolution::EstimateFusionCost` recursing through
`CalculateClassicWindowCost`), during splash boot calibration-warm, first time
the tp8 cases (h32-s4096 scale) went through that phase. The process dies with
SIGSEGV, which nothing in-process can catch -- so this script compiles each
graph of the warm path ONE AT A TIME, printing a marker before each, and the
last marker printed names the killer.

Phases per case, in boot order:
  reference     the generalized 4D-einsum closed form
  ref_bf16      same at bf16
  variant[i]    honest variants (blocked bf16, online softmax scan)
  selfstats     error_stats(ref, ref)  -- the coprime-strided q99 subsample,
                the prime suspect: a stride-N slice over a flat 16.7M-element
                leaf is exactly the strided-window form XLA rewrites as a
                convolution
  tolerance     full calibrated_tolerance (reference_bf16 + variants + stats)
"""

from __future__ import annotations

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)  # ENTRY convention; unused
    ap.add_argument("--task", default="splash_attention")
    ap.add_argument("--cases", default=(
        "probe-h8-s4096,probe-h4-s2048,probe-h16-s1024,probe-h8-s4096-d64,"
        "tp8-h32-s4096,tp8-gqa32x8-s4096,tp8-mqa-h32kv1-s4096,"
        "probe-holdout-h4-s2049,tp8-holdout-h32-s2049"))
    args = ap.parse_args()

    import jax

    from pallas_arena.judge.problems import get_problem
    from pallas_arena.judge.problems.base import error_stats

    p = get_problem(args.task)
    block = jax.block_until_ready

    for name in args.cases.split(","):
        case = p.case_by_name(name)
        pc = p.for_case(case)
        inputs = p.make_inputs(jax.random.PRNGKey(1), case)
        block(inputs)
        print(f"[bisect] {name}: reference ...", flush=True)
        ref = pc.reference(*inputs)
        block(ref)
        print(f"[bisect] {name}: reference_bf16 ...", flush=True)
        block(pc.reference_bf16(*inputs))
        for i, v in enumerate(pc.honest_variants()):
            print(f"[bisect] {name}: variant[{i}] ...", flush=True)
            block(v(*inputs))
        print(f"[bisect] {name}: selfstats ...", flush=True)
        error_stats(ref, ref)
        print(f"[bisect] {name}: tolerance ...", flush=True)
        pc.calibrated_tolerance(inputs, ref)
        print(f"[bisect] {name}: OK", flush=True)
        del inputs, ref
        import gc

        gc.collect()
    print("[bisect] ALL PHASES SURVIVED", flush=True)


def probe_crasher() -> None:
    """Focused matrix on the identified killer: _honest_faithful_bf16 at
    h16-s1024 (jobs 3714344 / 3714519). Alternates run BEFORE the known
    crasher, so every surviving configuration prints its marker even though
    the crasher takes the process down. Distinguishes:
      block_q     -- is it the (h16, block 512, s1024) geometry?
      fp32 inputs -- is it the MIXED-PRECISION dot (bf16 in, f32 accum)?
      python loop -- is it the fused scan body?
    """
    import jax
    import jax.numpy as jnp

    from pallas_arena.judge.problems import get_problem
    from pallas_arena.judge.problems.splash_attention import _honest_faithful_bf16

    p = get_problem("splash_attention")
    case = p.case_by_name("probe-h16-s1024")
    q, k, v, seg = p.make_inputs(jax.random.PRNGKey(1), case)
    block = jax.block_until_ready

    print("[matrix] block_q=256 ...", flush=True)
    block(_honest_faithful_bf16(q, k, v, seg, block_q=256))
    print("[matrix] block_q=1024 (single block, no scan) ...", flush=True)
    block(_honest_faithful_bf16(q, k, v, seg, block_q=1024))
    print("[matrix] fp32 inputs, block_q=512 ...", flush=True)
    block(_honest_faithful_bf16(q.astype(jnp.float32), k.astype(jnp.float32),
                                v.astype(jnp.float32), seg, block_q=512))
    print("[matrix] h8 slice, block_q=512 ...", flush=True)
    block(_honest_faithful_bf16(q[:8], k[:8], v[:8], seg, block_q=512))
    print("[matrix] DEFAULT (the known crasher): bf16, block_q=512, h16 ...", flush=True)
    block(_honest_faithful_bf16(q, k, v, seg, block_q=512))
    print("[matrix] crasher SURVIVED?!", flush=True)


if __name__ == "__main__":
    import sys as _sys

    if "--matrix" in _sys.argv:
        _sys.argv.remove("--matrix")
        probe_crasher()
    else:
        main()
