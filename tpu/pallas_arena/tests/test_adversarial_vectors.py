"""Adversarial vector library tests (tolerance-exploitation defense).

Two things are proven here:
  1. every task's adversarial generators produce the intended structure and
     the REFERENCE handles them (fully-masked rows == 0 not NaN, a->1 stays
     finite, empty expert groups fine, ...);
  2. the per-element error-TAIL check has teeth: a silent approximator that
     looks fine on Gaussian inputs is decisively caught on the adversarial
     vectors (the attack class a global allclose can never catch).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pallas_arena.judge.problems import get_problem, problem_names
from pallas_arena.judge.problems.base import (
    check_tolerance,
    error_stats,
    tolerance_from_reference,
)

KEY = jax.random.PRNGKey(11)


@pytest.mark.parametrize("name", problem_names())
def test_adversarial_vectors_reference_contract(name):
    """Generators build; expectations hold on the reference; calibrated
    tolerances are finite; the reference passes its own tolerance."""
    p = get_problem(name)
    advs = p.adversarial_cases()
    assert advs, f"{name} must ship adversarial vectors"
    for i, adv in enumerate(advs):
        inputs = adv.make_inputs(jax.random.fold_in(KEY, i))
        ref = p.reference(*inputs)
        if adv.expect is not None:
            adv.expect(ref, inputs)
        tol = tolerance_from_reference(ref, p.reference_bf16(*inputs))
        assert np.isfinite([tol["max"], tol["q99"]]).all()
        ok, why = check_tolerance(error_stats(ref, ref), tol)
        assert ok, why


def _adv_by_name(problem, name):
    for adv in problem.adversarial_cases():
        if adv.name == name:
            return adv
    raise KeyError(name)


def test_rg_lru_a_to_one_calibration_and_reset_vector_teeth():
    """RG-LRU: (a) the a->1 long-memory vector stays discriminative — the
    fp32 reference passes its own calibrated tolerance there (the gates are
    fp32 inputs; calibration must NOT quantize them, see reference_bf16);
    (b) a kernel that silently ignores `reset` is decisively caught by the
    dense-reset vector."""
    p = get_problem("rg_lru")

    adv1 = _adv_by_name(p, "a-to-one-long-memory")
    inputs1 = adv1.make_inputs(jax.random.PRNGKey(21))
    ref1 = p.reference(*inputs1)
    adv1.expect(ref1, inputs1)
    tol1 = tolerance_from_reference(ref1, p.reference_bf16(*inputs1))
    assert tol1["max"] < 0.1, "a->1 calibration collapsed (gates must not be bf16-quantized)"
    ok, why = check_tolerance(error_stats(ref1, ref1), tol1)
    assert ok, why

    def reset_ignoring_kernel(x, a, reset):
        return p.reference(x, a, jnp.zeros_like(reset))

    adv2 = _adv_by_name(p, "dense-reset-boundaries")
    inputs2 = adv2.make_inputs(jax.random.PRNGKey(22))
    ref2 = p.reference(*inputs2)
    tol2 = tolerance_from_reference(ref2, p.reference_bf16(*inputs2))
    stats = error_stats(reset_ignoring_kernel(*inputs2), ref2)
    ok, _ = check_tolerance(stats, tol2)
    assert not ok, "reset-ignoring kernel must fail on the dense-reset vector"


def test_error_tails_catch_unsafe_softmax_on_masked_rows():
    """Splash: a no-max-shift softmax with no masked-row guard NaNs on
    fully-masked rows; the expectation `expect_masked_zero` and the finite
    check both catch it."""
    p = get_problem("splash_attention")

    def unsafe_kernel(q, k, v, segment_ids):
        q32, k32, v32 = (t.astype(jnp.float32) for t in (q, k, v))
        s = q.shape[1]
        idx = jnp.arange(s)
        mask = (
            (idx[:, None] >= idx[None, :])
            & (segment_ids[:, None] == segment_ids[None, :])
            & (segment_ids != 0)[None, :]
            & (segment_ids != 0)[:, None]
        )
        logits = jnp.einsum("hqd,hkd->hqk", q32, k32)
        logits = jnp.where(mask[None], logits, -jnp.inf)
        p_ = jnp.exp(logits)  # no max shift
        p_ = p_ / p_.sum(-1, keepdims=True)  # 0/0 on dead rows -> NaN
        return jnp.einsum("hqk,hkd->hqd", p_, v32)

    adv = _adv_by_name(p, "fully-masked-rows")
    inputs = adv.make_inputs(jax.random.PRNGKey(23))
    ref = p.reference(*inputs)
    adv.expect(ref, inputs)  # the reference is clean
    stats = error_stats(unsafe_kernel(*inputs), ref)
    assert not stats["finite"], "unsafe softmax must NaN on masked rows"
    tol = tolerance_from_reference(ref, p.reference_bf16(*inputs))
    ok, _ = check_tolerance(stats, tol)
    assert not ok


def test_error_tails_catch_empty_group_skipper():
    """Megablox: a kernel whose row bookkeeping is wrong when a group is
    empty passes on dense uniform sizes, fails on the empty-group vector."""
    p = get_problem("megablox_gmm")

    def buggy_kernel(lhs, rhs, sizes):
        # bug: assumes every group non-empty — indexes groups by nonzero
        # order, so after an empty group every block uses the WRONG expert
        lhs32, rhs32 = lhs.astype(jnp.float32), rhs.astype(jnp.float32)
        sizes_n = np.asarray(sizes)
        used = [g for g in range(len(sizes_n)) if sizes_n[g] > 0]
        out = []
        for j, g in enumerate(used):
            n = int(sizes_n[used[j]])
            start = int(sizes_n[:g].sum())
            out.append(lhs32[start : start + n] @ rhs32[j])  # rhs[j] != rhs[g]!
        return jnp.concatenate(out, axis=0)

    tiny = p.case_by_name("tiny")
    rand_inputs = p.make_inputs(jax.random.PRNGKey(24), tiny)  # all non-empty
    ref_r = p.reference(*rand_inputs)
    stats_rand = error_stats(buggy_kernel(*rand_inputs), ref_r)
    tol_r = tolerance_from_reference(ref_r, p.reference_bf16(*rand_inputs))
    ok_rand, _ = check_tolerance(stats_rand, tol_r)
    assert ok_rand, "bug is invisible while every group is non-empty"

    adv = _adv_by_name(p, "empty-expert-groups")
    adv_inputs = adv.make_inputs(jax.random.PRNGKey(25))
    ref_a = p.reference(*adv_inputs)
    tol_a = tolerance_from_reference(ref_a, p.reference_bf16(*adv_inputs))
    ok_adv, _ = check_tolerance(error_stats(buggy_kernel(*adv_inputs), ref_a), tol_a)
    assert not ok_adv, "empty-group vector must expose the bookkeeping bug"


def test_flce_label_in_tail_catches_topk_lse():
    """FLCE: a top-k-truncated LSE looks tolerable on average but its error
    TAIL blows past the calibrated q99/max budget on structured inputs."""
    p = get_problem("flce")

    def topk_kernel(hidden, w, targets, k=32):
        logits = hidden.astype(jnp.float32) @ w.astype(jnp.float32)
        tl = jnp.take_along_axis(logits, targets[:, None], axis=1)[:, 0]
        top = jax.lax.top_k(logits, k)[0]
        lse = jax.nn.logsumexp(top, axis=-1)  # truncated tail
        return tl - lse

    adv = _adv_by_name(p, "label-in-tail")
    inputs = adv.make_inputs(jax.random.PRNGKey(26))
    ref = p.reference(*inputs)
    tol = tolerance_from_reference(ref, p.reference_bf16(*inputs))
    stats = error_stats(topk_kernel(*inputs), ref)
    ok, _ = check_tolerance(stats, tol)
    assert not ok, "truncated-LSE approximator must fail the tail check"


def test_rpa_single_token_and_page_boundary_vectors():
    p = get_problem("ragged_paged_attention")
    adv = _adv_by_name(p, "single-token-decode")
    inputs = adv.make_inputs(jax.random.PRNGKey(27))
    adv.expect(p.reference(*inputs), inputs)  # closed form: out == v[first]
    adv2 = _adv_by_name(p, "page-boundary-lengths")
    inputs2 = adv2.make_inputs(jax.random.PRNGKey(28))
    ps = inputs2[1].shape[1]
    assert (np.asarray(inputs2[4]) % ps == 0).all()
    adv2.expect(p.reference(*inputs2), inputs2)


def test_rmsnorm_zero_row_and_overflow_vectors():
    p = get_problem("rmsnorm")
    for name in ("zero-row", "near-overflow-bf16", "outlier-rows"):
        adv = _adv_by_name(p, name)
        inputs = adv.make_inputs(jax.random.PRNGKey(29))
        adv.expect(p.reference(*inputs), inputs)
