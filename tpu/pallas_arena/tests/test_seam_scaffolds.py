"""The rf3s scaffolds must be API-perfect BEFORE any model sees them.

A wiring bug in a scaffold becomes every candidate's bug, so the bar here is
the real judge machinery, not eyeballing: compose each scaffold with its
naive test-only fill and require that the result

  1. passes the AOT pre-gate (AST incl. the pallas_call requirement, poison
     stubs, export at every declared shape -- the exact validity gate),
  2. is numerically CORRECT vs the reference at the tiny smoke cases within
     the calibrated band,
  3. (rg_lru, whose backward fill exists) differentiates through the
     custom_vjp wiring and matches the reference gradients per leaf.

The naive fills read the whole padded sequence per block -- CPU-fine,
deliberately not TPU-viable at production shapes; they prove WIRING.
"""

from __future__ import annotations

import os

import pytest

os.environ["PALLAS_INTERPRET"] = "1"  # before any scaffold program is exec'd
# Mosaic-targeting exports mmap more address space than plain-JAX children:
# the 16GB RLIMIT_AS default SIGABRTs the export child on any real
# pallas_call (bisected: novjp+rlimit16 -> rc=-6, novjp+rlimit64 -> clean).
os.environ["ARENA_RLIMIT_GB"] = "64"

from pallas_arena.judge.problems import get_problem
from pallas_arena.judge.problems.base import check_tolerance, error_stats
from pallas_arena.probe.seam_scaffolds import (
    RGLRU_NAIVE_BWD,
    RGLRU_NAIVE_FWD,
    RGLRU_SCAFFOLD,
    SPLASH_NAIVE_FWD,
    SPLASH_SCAFFOLD,
    compose,
)


def _load_kernel(program: str):
    ns: dict = {}
    exec(compile(program, "<scaffold>", "exec"), ns)
    return ns["kernel"]


@pytest.fixture(scope="module")
def splash_prog():
    return compose(SPLASH_SCAFFOLD, {"YOUR FORWARD BODY": SPLASH_NAIVE_FWD})


@pytest.fixture(scope="module")
def rglru_prog():
    return compose(RGLRU_SCAFFOLD, {
        "YOUR FORWARD BODY": RGLRU_NAIVE_FWD,
        "YOUR BACKWARD BODY": RGLRU_NAIVE_BWD,
    })


def _export_for_tpu(task: str, program: str) -> dict:
    """The judge's REAL validity gate: AST + poison stubs + jax.export at every
    declared shape, targeting TPU from a CPU child under the device-kind shim.
    (The CPU-backend pregate SIGABRTs on any real pallas_call -- the battery
    never noticed because no battery candidate ever contained one.)"""
    from pallas_arena.judge import grader
    from pallas_arena.judge.problems import get_problem
    from pallas_arena.judge.worker import build_signatures

    p = get_problem(task)
    scored = [c for c in p.shape_cases() if c.smoke and not c.holdout]
    holdout = [c for c in p.shape_cases() if c.smoke and c.holdout]
    sigs, _, _, _ = build_signatures(p, scored, holdout, [])
    return grader.grade(
        task, program, mode="aot_export", smoke=True,
        export_signatures=sigs, export_platforms=["tpu"],
        child_env={"JAX_PLATFORMS": "cpu"},
    )


def test_splash_scaffold_exports_for_tpu(splash_prog):
    r = _export_for_tpu("splash_attention", splash_prog)
    assert r.get("passed"), r.get("violations")


def test_rglru_scaffold_exports_for_tpu(rglru_prog):
    r = _export_for_tpu("rg_lru", rglru_prog)
    assert r.get("passed"), r.get("violations")


@pytest.mark.parametrize("case_name", ["tiny", "tiny-ragged"])
def test_splash_scaffold_is_correct_at_tiny(splash_prog, case_name):
    import jax

    p = get_problem("splash_attention")
    kfn = _load_kernel(splash_prog)
    case = p.case_by_name(case_name)
    ins = p.make_inputs(jax.random.PRNGKey(0), case)
    ref = p.reference(*ins)
    tol = p.calibrated_tolerance(ins, ref)
    ok, why = check_tolerance(error_stats(kfn(*ins), ref), tol)
    assert ok, f"{case_name}: {why}"


@pytest.mark.parametrize("case_name", ["tiny", "tiny-ragged"])
def test_rglru_scaffold_is_correct_at_tiny(rglru_prog, case_name):
    import jax

    p = get_problem("rg_lru")
    kfn = _load_kernel(rglru_prog)
    case = p.case_by_name(case_name)
    ins = p.make_inputs(jax.random.PRNGKey(0), case)
    ref = p.reference(*ins)
    tol = p.calibrated_tolerance(ins, ref)
    ok, why = check_tolerance(error_stats(kfn(*ins), ref), tol)
    assert ok, f"{case_name}: {why}"


def test_rglru_scaffold_backward_matches_reference(rglru_prog):
    """The custom_vjp wiring end-to-end: differentiate the composed program
    and compare per-leaf against the reference gradients -- the exact check
    the judge runs."""
    import jax

    from pallas_arena.judge.problems.base import (
        check_grad_tolerance,
        grad_leaf_tolerances,
    )

    p = get_problem("rg_lru")
    kfn = _load_kernel(rglru_prog)
    case = p.case_by_name("tiny")
    ins = p.make_inputs(jax.random.PRNGKey(1), case)
    ref_g = p.grad_outputs(lambda *i: p.reference(*i), *ins)
    cal = [p.grad_outputs(lambda *i: p.reference_bf16(*i), *ins)]
    for v in p.grad_calibration_variants():
        cal.append(p.grad_outputs(v, *ins))
    tols = grad_leaf_tolerances(ref_g, *cal)
    cand_g = p.grad_outputs(kfn, *ins)
    ok, why = check_grad_tolerance(cand_g, ref_g, tols)
    assert ok, why


def test_splash_scaffold_without_bwd_fill_still_exports_forward(splash_prog):
    """The backward bodies are unfilled in the shipped scaffold; the judge's
    scored-backward contract must let the forward still grade. The pregate
    lowers the grad functional and, under bwd_gates=False, records rather
    than rejects a non-differentiable candidate -- the scaffold with only
    the forward filled must therefore PASS."""
    # splash_prog has only the forward fill: its custom_vjp bwd raises
    # NotImplementedError at trace time -> exactly the recorded-not-rejected
    # path. Covered by test_splash_scaffold_passes_the_pregate above; this
    # test pins the INTENT so a future bwd_gates flip fails loudly here.
    p = get_problem("splash_attention")
    assert p.bwd_gates is False
