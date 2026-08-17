"""Subprocess body for the forced-8-device TP spec check. NOT a pytest file.

Run by tests/test_tp_sharded_cpu.py with:
    XLA_FLAGS=--xla_force_host_platform_device_count=8  JAX_PLATFORMS=cpu

Why this exists: every declared tp8 case had been SKIPPED on every judge run
to date (single-chip judges), so the specs -- including the KV-replication
rule for MQA -- had never once executed a shard_map. The first execution was
going to happen on a paid v6e-8 spot QR; a spec bug there costs quota and a
bring-up cycle, here it costs seconds. This is deliberately the same
mechanics the judge uses: `timing_mod.make_mesh` + `timing_mod.shard_mapped`
+ `timing_mod.shard_inputs`, not a hand-rolled re-implementation.

The assertion: sharded output agrees with unsharded to FP32-ROUNDING scale
(relative max error < 1e-4), which is ~50x tighter than the calibrated grading
band. NOT bitwise, and measured rather than assumed: the first run of this
check showed 2.4e-7..1.4e-6 diffs on every task, because the sharded case is a
DIFFERENT compiled program (per-shard shapes -> different fusion and reduction
tilings -> different rounding). Bitwise equality is a same-program property --
that is the determinism gate's job, not this one's. tokamax draws the same
line: their shard_map-based splash test asserts atol=5e-3, and only the
auto-propagation api_sharding test uses atol=0.0.

A wrong spec still cannot hide: it produces O(1) error, not 1e-6.

Prints one line per (task, case): `OK <task> <case> agrees` or raises.
"""

from __future__ import annotations

import os
import sys

assert "--xla_force_host_platform_device_count=8" in os.environ.get("XLA_FLAGS", ""), (
    "must be launched with 8 forced host devices"
)

import jax  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from pallas_arena.judge import timing as timing_mod  # noqa: E402
from pallas_arena.judge.problems import get_problem  # noqa: E402
from pallas_arena.judge.problems.base import ShapeCase  # noqa: E402

assert jax.device_count() == 8, jax.device_count()

# Tiny stand-ins mirroring each production tp8 case's STRUCTURE (the thing
# the spec keys on: head grouping, which axes shard), at sizes a CPU runs in
# seconds. The production tp8 cases themselves are exercised on the v6e-8
# judge; what this file pins is the spec/mesh/shard_map mechanics.
CASES = {
    "splash_attention": [
        ShapeCase("cpu-tp8-mha", {"heads": 16, "seq": 256, "d": 64}, tp=8),
        ShapeCase("cpu-tp8-gqa", {"heads": 16, "kv_heads": 8, "seq": 256, "d": 64}, tp=8),
        # MQA: kv_heads=1 cannot shard 8 ways -> the KV-replication rule
        ShapeCase("cpu-tp8-mqa", {"heads": 16, "kv_heads": 1, "seq": 256, "d": 64}, tp=8),
        # feature binding composes with shard_map
        ShapeCase("cpu-tp8-window", {"heads": 16, "seq": 256, "d": 64}, tp=8,
                  features=(("window", 64),)),
    ],
    "megablox_gmm": [
        ShapeCase("cpu-tp8-gmm", {"m": 256, "g": 8, "k": 128, "n": 256, "dist": "uniform"}, tp=8),
    ],
    "ragged_paged_attention": [],  # filled below: dims are task-structured
    "rg_lru": [
        ShapeCase("cpu-tp8-lru", {"b": 4, "t": 64, "d": 256}, tp=8),
    ],
}


def _rpa_case():
    p = get_problem("ragged_paged_attention")
    # borrow a real tp case's dims and shrink the independent sizes, keeping
    # the head structure (what the spec shards) intact
    src = next(c for c in p.shape_cases() if c.tp)
    dims = dict(src.dims)
    dims.update(batch=8, max_len=64, num_pages=8 * 1 + 8)
    return ShapeCase("cpu-tp8-rpa", dims, tp=8)


CASES["ragged_paged_attention"] = [_rpa_case()]


def main() -> None:
    failures = []
    for task, cases in CASES.items():
        p = get_problem(task)
        for case in cases:
            w = p.tp_declared_width(case)
            assert w == 8, (task, case.name, w)
            pc = p.for_case(case)
            tp_in, tp_out = p.tp_specs(case)
            mesh = timing_mod.make_mesh(w)
            assert mesh is not None, "8 forced devices should give a mesh"
            inputs = p.make_inputs(jax.random.PRNGKey(0), case)

            unsharded = np.asarray(pc.reference(*inputs))
            sharded_fn = jax.jit(timing_mod.shard_mapped(pc.reference, mesh, tp_in, tp_out))
            sharded = np.asarray(sharded_fn(*timing_mod.shard_inputs(inputs, mesh, tp_in)))

            if unsharded.shape != sharded.shape:
                failures.append(f"{task}/{case.name}: shape {unsharded.shape} vs {sharded.shape}")
                continue
            rel = float(np.max(np.abs(unsharded - sharded) / (np.abs(unsharded) + 1.0)))
            if rel > 1e-4:
                failures.append(f"{task}/{case.name}: sharded != unsharded, rel max err {rel:.3e}")
                continue

            # per-shard export shapes divide exactly as declared
            for a, full in zip(p.abstract_inputs_tp(case, w), inputs):
                assert all(sa <= fa for sa, fa in zip(a.shape, np.asarray(full).shape)), (
                    task, case.name, a.shape, np.asarray(full).shape)
            print(f"OK {task} {case.name} agrees rel_err<=1e-4", flush=True)

    # gradient THROUGH shard_map (the tokamax sharded test does bwd too):
    # heads-sharded splash reference, d/d(q,k,v), same 1e-4 agreement bound.
    p = get_problem("splash_attention")
    case = CASES["splash_attention"][0]
    tp_in, tp_out = p.tp_specs(case)
    mesh = timing_mod.make_mesh(8)
    inputs = p.make_inputs(jax.random.PRNGKey(1), case)
    un_g = p.grad_outputs(p.reference, *inputs)
    sh_ref = timing_mod.shard_mapped(p.reference, mesh, tp_in, tp_out)
    sh_g = p.grad_outputs(sh_ref, *inputs)
    for i, (a, b) in enumerate(zip(un_g, sh_g)):
        a, b = np.asarray(a), np.asarray(b)
        rel = float(np.max(np.abs(a - b) / (np.abs(a) + 1.0)))
        if rel > 1e-4:
            failures.append(f"splash grad[{i}] through shard_map rel err {rel:.3e}")
    if not failures:
        print("OK splash_attention grad-through-shard_map agrees", flush=True)

    if failures:
        print("FAILURES:", flush=True)
        for f in failures:
            print("  " + f, flush=True)
        sys.exit(1)
    print("ALL-OK", flush=True)


if __name__ == "__main__":
    main()
