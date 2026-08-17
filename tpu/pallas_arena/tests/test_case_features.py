"""Static per-case features: sliding window, logit soft-cap.

WHY THESE EXIST. tokamax parameterizes its TPU attention test over
`logits_soft_cap=[None, 3.4]` and ships `LocalMask` for windowed attention,
because production models use both (Mistral/Ministral/Gemma-2 window;
Gemma-2/3 soft-cap). Our contract graded only the plain causal path, so a
candidate implementing just that path was being compared against production
kernels that support all of it -- and scored as if the comparison were fair.

The tests below pin the two properties that make feature grading trustworthy:

  1. SEMANTICS -- the features mean what the production kernels mean, checked
     against limits with known closed forms rather than golden numbers.
  2. WHOLE-FAMILY BINDING -- the failure mode that would quietly destroy the
     tolerance band is PARTIAL binding (windowed reference vs unwindowed honest
     variants), which inflates the band to the windowed-vs-full gap and admits
     almost anything. `Problem.for_case` binds the family together; these
     tests fail if that ever regresses to per-callable binding.
"""

from __future__ import annotations

import numpy as np
import pytest

from pallas_arena.judge.problems import get_problem
from pallas_arena.judge.problems.base import ShapeCase, error_stats
from pallas_arena.judge.problems.splash_attention import (
    _xla_grouped_attention,
    causal_segment_attention,
)


@pytest.fixture(scope="module")
def fixture():
    import jax

    p = get_problem("splash_attention")
    case = p.case_by_name("tiny") if _has(p, "tiny") else p.shape_cases()[0]
    return p, case, p.make_inputs(jax.random.PRNGKey(0), case)


def _has(problem, name):
    return any(c.name == name for c in problem.shape_cases())


# ------------------------------------------------------------------ semantics
def test_window_at_least_seq_is_plain_causal(fixture):
    """A window wider than the sequence cannot mask anything, so it must be
    bit-comparable to plain causal attention. Catches an off-by-one in the
    window predicate, which would otherwise show up only as a small numeric
    drift that a calibrated band might absorb."""
    _, _, ins = fixture
    seq = ins[0].shape[1]
    plain = np.asarray(causal_segment_attention(*ins))
    wide = np.asarray(causal_segment_attention(*ins, window=seq + 1))
    np.testing.assert_allclose(plain, wide, atol=1e-6)


def test_window_one_reduces_to_the_value_vector(fixture):
    """window=1 lets each query see only itself; softmax over a single key is
    exactly 1, so the output must equal v on live rows. This is the strongest
    available closed form for the window semantics -- it pins the interval as
    (i-window, i] (inclusive of self) rather than [i-window, i)."""
    import jax.numpy as jnp

    _, _, ins = fixture
    q, k, v, seg = ins
    out = np.asarray(causal_segment_attention(q, k, v, seg, window=1))
    live = np.asarray(seg) != 0
    np.testing.assert_allclose(
        out[:, live, :], np.asarray(v.astype(jnp.float32))[:, live, :], atol=2e-2
    )


def test_window_actually_masks(fixture):
    """Guards against a window that is silently ignored -- which would make
    every windowed case a duplicate of the plain one while reporting coverage
    we do not have."""
    _, _, ins = fixture
    plain = np.asarray(causal_segment_attention(*ins))
    narrow = np.asarray(causal_segment_attention(*ins, window=8))
    assert not np.allclose(plain, narrow)


def test_soft_cap_is_a_noop_in_the_limit(fixture):
    """cap * tanh(x / cap) -> x as cap -> inf. Pins the formula itself, not a
    golden value."""
    _, _, ins = fixture
    plain = np.asarray(causal_segment_attention(*ins))
    capped = np.asarray(causal_segment_attention(*ins, soft_cap=1e6))
    np.testing.assert_allclose(plain, capped, atol=1e-4)


def test_soft_cap_applies_before_masking(fixture):
    """Order is load-bearing: capping AFTER the mask would squash the -inf
    sentinel to +-cap and unmask every forbidden position, which shows up as
    padding rows that are no longer exactly zero."""
    _, _, ins = fixture
    q, k, v, seg = ins
    out = np.asarray(causal_segment_attention(q, k, v, seg, soft_cap=3.4))
    dead = np.asarray(seg) == 0
    if dead.any():
        assert np.all(out[:, dead, :] == 0.0)


@pytest.mark.parametrize(
    "feats",
    [{"window": 64}, {"soft_cap": 3.4}, {"window": 64, "soft_cap": 3.4}],
)
def test_grouped_baseline_agrees_under_features(fixture, feats):
    """The denominator must compute the SAME function as the reference under
    every feature combination -- otherwise the ratio is between two different
    problems and the reward is meaningless."""
    p, _, ins = fixture
    ref = causal_segment_attention(*ins, **feats)
    got = _xla_grouped_attention(*ins, **feats)
    assert error_stats(got, ref)["max"] < 1e-4


# -------------------------------------------------------- whole-family binding
def test_for_case_without_features_is_the_same_object(fixture):
    """The featureless path must keep its exact identity: no copy, no extra
    partials, no behaviour change for any existing case."""
    p, case, _ = fixture
    assert p.for_case(case) is p


def test_for_case_binds_reference_and_variants_together(fixture):
    """THE regression guard. A view whose reference is windowed but whose
    honest variants are not would produce a band ~the size of the
    windowed-vs-full difference. Compare against the band computed with a
    consistently-bound family: it must not blow up."""
    p, case, ins = fixture
    wcase = ShapeCase("w", case.dims, features=(("window", 64),))
    view = p.for_case(wcase)

    ref_w = view.reference(*ins)
    band_w = view.calibrated_tolerance(ins, ref_w)
    band_plain = p.calibrated_tolerance(ins, p.reference(*ins))

    # A mis-bound family shows up as orders of magnitude, not percent.
    assert band_w["max"] < 10 * band_plain["max"], (band_w, band_plain)

    # And the variants really are windowed: an unbound variant would differ
    # from the windowed reference by far more than the band allows.
    for variant in view.honest_variants():
        assert error_stats(variant(*ins), ref_w)["max"] <= band_w["max"]


def test_for_case_binds_baseline_candidates(fixture):
    """Baseline election happens per case; if candidates were left unbound the
    elected denominator would compute plain attention for a windowed case."""
    p, case, ins = fixture
    wcase = ShapeCase("w", case.dims, features=(("window", 64),))
    view = p.for_case(wcase)
    ref_w = view.reference(*ins)
    got = view.baseline_candidates()["xla-grouped"](*ins)
    assert error_stats(got, ref_w)["max"] < 1e-4


# ------------------------------------------------- grouped attention under TP
def test_tp_replicates_kv_when_it_cannot_be_sharded():
    """MQA over a mesh: the KV head axis is 1 and cannot be split 8 ways.

    tokamax's `test_broadcasted_multi_query_attention` is exactly this case --
    a single KV head run under partitioning -- and the name states the rule:
    the short axis broadcasts instead of splitting. Sharding it would fail at
    best and silently grade a wrong program at worst.
    """
    from jax.sharding import PartitionSpec as P

    p = get_problem("splash_attention")
    mqa = ShapeCase("m", {"heads": 32, "kv_heads": 1, "seq": 1024, "d": 128}, tp=8)
    (q_s, k_s, v_s, seg_s), out_s = p.tp_specs(mqa)
    assert q_s == P("tp", None, None), q_s
    assert k_s == P(None, None, None), k_s
    assert v_s == P(None, None, None), v_s
    assert out_s == P("tp", None, None)


def test_tp_still_shards_kv_when_it_divides():
    """MHA and GQA with kv_heads % width == 0 must keep sharding KV --
    replicating it there would multiply KV traffic by the mesh width and make
    the denominator artificially slow."""
    from jax.sharding import PartitionSpec as P

    p = get_problem("splash_attention")
    for dims in (
        {"heads": 32, "seq": 1024, "d": 128},                    # MHA: kv == 32
        {"heads": 32, "kv_heads": 8, "seq": 1024, "d": 128},     # GQA: 8 % 8 == 0
    ):
        (_, k_s, v_s, _), _ = p.tp_specs(ShapeCase("c", dims, tp=8))
        assert k_s == P("tp", None, None), (dims, k_s)
        assert v_s == P("tp", None, None), (dims, v_s)


def test_tp_declared_width_accepts_mqa_that_would_otherwise_be_indivisible():
    """`tp_declared_width` raises when a declared width does not divide a
    SHARDED axis. With KV replicated for MQA there is no longer such an axis,
    so the case must validate rather than raise -- this is the end-to-end
    consequence of the rule above."""
    p = get_problem("splash_attention")
    mqa = ShapeCase("m", {"heads": 32, "kv_heads": 1, "seq": 1024, "d": 128}, tp=8)
    assert p.tp_declared_width(mqa) == 8

    shard_shapes = [tuple(a.shape) for a in p.abstract_inputs_tp(mqa, 8)]
    assert shard_shapes[0][0] == 4, shard_shapes   # q heads split 32 -> 4
    assert shard_shapes[1][0] == 1, shard_shapes   # kv head replicated, stays 1


def test_features_change_the_exported_signature_identity():
    """Two cases with IDENTICAL shapes but different features must export as
    two artifacts. Deduping on shapes alone would grade the windowed case with
    the plain kernel and report a pass it never earned."""
    from pallas_arena.judge.worker import build_signatures

    p = get_problem("splash_attention")
    base = p.shape_cases()[0]
    plain = ShapeCase("plain", base.dims)
    windowed = ShapeCase("windowed", base.dims, features=(("window", 64),))

    sigs, case_sig, _, _ = build_signatures(p, [plain, windowed], [], [])
    assert case_sig["plain"] != case_sig["windowed"], sigs
    by_name = {s["name"]: s for s in sigs}
    assert by_name[case_sig["windowed"]]["features"] == {"window": 64}
    assert "features" not in by_name[case_sig["plain"]]
