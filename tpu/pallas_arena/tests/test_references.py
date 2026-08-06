"""Layer-1 tests: every reference vs an independent closed form on tiny
shapes (CPU), plus registry/spec sanity for all six tasks."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pallas_arena.judge.problems import get_problem, problem_names
from pallas_arena.judge.problems.megablox_gmm import gmm_loop_reference
from pallas_arena.judge.problems.rg_lru import rg_lru_associative

ALL = problem_names()


def test_registry_complete():
    assert set(ALL) == {"rmsnorm", "splash_attention", "ragged_paged_attention",
                        "megablox_gmm", "flce", "rg_lru"}


@pytest.mark.parametrize("name", ALL)
def test_spec_shape_contract(name):
    p = get_problem(name)
    cases = p.shape_cases()
    assert any(c.holdout and not c.smoke for c in cases), "needs a holdout shape"
    assert any(c.smoke and not c.holdout for c in cases), "needs smoke cases"
    assert any(c.smoke and c.holdout for c in cases), "needs a smoke holdout"
    names = [c.name for c in cases]
    assert len(names) == len(set(names))


@pytest.mark.parametrize("name", ALL)
def test_reference_runs_and_is_finite_on_smoke(name):
    p = get_problem(name)
    for case in p.scored_cases(smoke=True):
        inputs = p.make_inputs(jax.random.PRNGKey(7), case)
        out = p.reference(*inputs)
        leaves = out if isinstance(out, (tuple, list)) else (out,)
        for leaf in leaves:
            assert np.isfinite(np.asarray(leaf, np.float64)).all()


@pytest.mark.parametrize("name", ALL)
def test_abstract_inputs_for_pregate(name):
    p = get_problem(name)
    case = p.scored_cases(smoke=True)[0]
    abstract = p.abstract_inputs(case)
    concrete = p.make_inputs(jax.random.PRNGKey(3), case)
    for a, c in zip(abstract, concrete):
        assert a.shape == c.shape and a.dtype == c.dtype


def test_rmsnorm_reference_vs_numpy():
    p = get_problem("rmsnorm")
    x, g = p.make_inputs(jax.random.PRNGKey(0), p.case_by_name("tiny"))
    got = np.asarray(p.reference(x, g))
    xn = np.asarray(x, np.float32)
    gn = np.asarray(g, np.float32)
    want = xn / np.sqrt((xn**2).mean(-1, keepdims=True) + 1e-6) * gn
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)


def test_flce_reference_vs_log_softmax_and_baseline():
    p = get_problem("flce")
    for cname in ("tiny", "tiny-ragged"):
        h, w, t = p.make_inputs(jax.random.PRNGKey(1), p.case_by_name(cname))
        ref = np.asarray(p.reference(h, w, t))
        logits = np.asarray(h, np.float32) @ np.asarray(w, np.float32)
        lsm = logits - jax.nn.logsumexp(jnp.asarray(logits), axis=-1,
                                        keepdims=True)
        want = np.take_along_axis(np.asarray(lsm), np.asarray(t)[:, None],
                                  axis=1)[:, 0]
        np.testing.assert_allclose(ref, want, rtol=1e-4, atol=1e-4)
        # our custom_vjp kernel (the baseline-to-beat) agrees at its own
        # bf16-logits precision (also covers the non-tile-divisible pad path)
        base = np.asarray(p.baseline(h, w, t))
        np.testing.assert_allclose(base, ref, rtol=2e-2, atol=2e-2)


def test_rg_lru_reference_vs_python_loop_and_associative():
    p = get_problem("rg_lru")
    x, a, reset = p.make_inputs(jax.random.PRNGKey(2),
                                p.case_by_name("tiny-ragged"))
    ref = np.asarray(p.reference(x, a, reset))
    xn = np.asarray(x, np.float32)
    an = np.asarray(a, np.float32) * (1.0 - np.asarray(reset, np.float32)[..., None])
    want = np.zeros_like(xn)
    h = np.zeros((xn.shape[0], xn.shape[2]), np.float32)
    for t in range(xn.shape[1]):
        h = an[:, t] * h + np.sqrt(np.maximum(1 - an[:, t] ** 2, 0)) * xn[:, t]
        want[:, t] = h
    np.testing.assert_allclose(ref, want, rtol=1e-4, atol=1e-4)
    assoc = np.asarray(rg_lru_associative(x, a, reset))
    np.testing.assert_allclose(assoc, ref, rtol=1e-4, atol=1e-4)


def test_splash_reference_vs_numpy_loop():
    p = get_problem("splash_attention")
    q, k, v, seg = p.make_inputs(jax.random.PRNGKey(4),
                                 p.case_by_name("tiny-ragged"))
    ref = np.asarray(p.reference(q, k, v, seg))
    qn, kn, vn = (np.asarray(t, np.float32) for t in (q, k, v))
    segn = np.asarray(seg)
    H, S, _ = qn.shape
    want = np.zeros_like(ref)
    for h in range(H):
        for i in range(S):
            if segn[i] == 0:
                continue
            js = [j for j in range(S)
                  if j <= i and segn[j] == segn[i] and segn[j] != 0]
            logits = np.array([qn[h, i] @ kn[h, j] for j in js])
            e = np.exp(logits - logits.max())
            want[h, i] = (e / e.sum()) @ vn[h, js]
    np.testing.assert_allclose(ref, want, rtol=2e-4, atol=2e-4)
    # padding rows exactly zero
    assert np.abs(ref[:, segn == 0, :]).max() == 0.0


def test_rpa_reference_vs_numpy_loop():
    p = get_problem("ragged_paged_attention")
    q, kp, vp, pt, sl = p.make_inputs(jax.random.PRNGKey(5),
                                      p.case_by_name("tiny"))
    ref = np.asarray(p.reference(q, kp, vp, pt, sl))
    qn = np.asarray(q, np.float32)
    kpn = np.asarray(kp, np.float32)
    vpn = np.asarray(vp, np.float32)
    ptn, sln = np.asarray(pt), np.asarray(sl)
    b, qh, d = qn.shape
    ps, kvh = kpn.shape[1], kpn.shape[2]
    group = qh // kvh
    for i in range(b):
        kk = kpn[ptn[i]].reshape(-1, kvh, d)[: sln[i]]
        vv = vpn[ptn[i]].reshape(-1, kvh, d)[: sln[i]]
        for hq in range(qh):
            hk = hq // group
            logits = kk[:, hk] @ qn[i, hq]
            e = np.exp(logits - logits.max())
            want = (e / e.sum()) @ vv[:, hk]
            np.testing.assert_allclose(ref[i, hq], want, rtol=2e-4, atol=2e-4)


def test_megablox_reference_vs_loop_including_empty_and_skew():
    p = get_problem("megablox_gmm")
    for cname in ("tiny", "tiny-zipf"):
        lhs, rhs, sizes = p.make_inputs(jax.random.PRNGKey(6),
                                        p.case_by_name(cname))
        assert int(np.asarray(sizes).sum()) == lhs.shape[0]
        ref = np.asarray(p.reference(lhs, rhs, sizes))
        want = gmm_loop_reference(lhs, rhs, sizes)
        np.testing.assert_allclose(ref, want, rtol=2e-4, atol=2e-4)
    # empty groups + max skew agree with the loop form too
    lhs, rhs, _ = p.make_inputs(jax.random.PRNGKey(8), p.case_by_name("tiny"))
    m = lhs.shape[0]
    for sizes in ([0, m // 2, 0, m - m // 2], [m, 0, 0, 0]):
        s = jnp.asarray(sizes, jnp.int32)
        np.testing.assert_allclose(
            np.asarray(p.reference(lhs, rhs, s)),
            gmm_loop_reference(lhs, rhs, s), rtol=2e-4, atol=2e-4)


def test_megablox_group_size_sampling_uniform_vs_zipf():
    p = get_problem("megablox_gmm")
    _, _, uni = p.make_inputs(jax.random.PRNGKey(9), p.case_by_name("tiny"))
    _, _, zipf = p.make_inputs(jax.random.PRNGKey(9),
                               p.case_by_name("tiny-zipf"))
    assert int(np.asarray(uni).sum()) == int(np.asarray(zipf).sum()) == 64
    # zipf must be visibly skewed, uniform must not be
    assert np.asarray(zipf).max() > np.asarray(uni).max()
