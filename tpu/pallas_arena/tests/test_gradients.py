"""Gradient-contract tests for the fwd+bwd tasks (RMSNorm, FLCE):
reference grads vs finite differences, and the baseline kernel's custom_vjp
vs the reference grads at its own calibrated tolerance."""

import jax
import jax.numpy as jnp
import numpy as np

from pallas_arena.judge.problems import get_problem
from pallas_arena.judge.problems.base import (
    check_tolerance,
    error_stats,
    tolerance_from_reference,
)


def _fd_check(scalar_fn, x, coords, eps=1e-3, rtol=0.08):
    """Central finite differences on a few coordinates vs jax.grad."""
    g = jax.grad(scalar_fn)(x)
    for idx in coords:
        xp = x.at[idx].add(eps)
        xm = x.at[idx].add(-eps)
        fd = (scalar_fn(xp) - scalar_fn(xm)) / (2 * eps)
        got = np.asarray(g)[idx]
        assert np.isfinite(fd) and np.isfinite(got)
        denom = max(abs(fd), abs(got), 1e-3)
        assert abs(fd - got) / denom < rtol, (idx, fd, got)


def test_rmsnorm_reference_grad_vs_finite_differences():
    p = get_problem("rmsnorm")
    x, gamma = p.make_inputs(jax.random.PRNGKey(31), p.case_by_name("tiny"))
    x32 = x.astype(jnp.float32)
    probe = jnp.cos(jnp.arange(x.size, dtype=jnp.float32)).reshape(x.shape)

    def scalar_x(xv):
        return jnp.sum(p.reference(xv.astype(x.dtype), gamma) * probe)

    def scalar_g(gv):
        return jnp.sum(p.reference(x, gv) * probe)

    # NB: fd through the bf16 cast needs a large eps; use fp32 path directly
    def scalar_x32(xv):
        var = jnp.mean(jnp.square(xv), axis=-1, keepdims=True)
        return jnp.sum(xv * jax.lax.rsqrt(var + 1e-6) * gamma * probe)

    _fd_check(scalar_x32, x32, [(0, 0), (3, 5), (17, 63)])
    _fd_check(scalar_g, gamma, [(0,), (5,), (63,)])


def test_flce_baseline_grad_matches_reference_grad():
    """Our custom_vjp kernel's d/d(hidden) == the dense log_softmax grad at
    the kernel's own calibrated (bf16-logits) tolerance, on the divisible
    AND the padded (non-tile-divisible) path."""
    p = get_problem("flce")
    for cname in ("tiny", "tiny-ragged"):
        inputs = p.make_inputs(jax.random.PRNGKey(32), p.case_by_name(cname))
        ref_g = p.grad_outputs(lambda *i: p.reference(*i), *inputs)
        cal_g = p.grad_outputs(lambda *i: p.reference_bf16(*i), *inputs)
        base_g = p.grad_outputs(lambda *i: p.baseline(*i), *inputs)
        tol = tolerance_from_reference(ref_g, cal_g)
        ok, why = check_tolerance(error_stats(base_g, ref_g), tol)
        assert ok, f"{cname}: {why}"


def test_flce_reference_grad_vs_finite_differences():
    p = get_problem("flce")
    hidden, w, targets = p.make_inputs(jax.random.PRNGKey(33),
                                       p.case_by_name("tiny"))
    h32 = hidden.astype(jnp.float32)
    n = h32.shape[0]
    probe = jnp.cos(jnp.arange(n, dtype=jnp.float32))

    def scalar_h(hv):
        logits = hv @ w.astype(jnp.float32)
        tl = jnp.take_along_axis(logits, targets[:, None], axis=1)[:, 0]
        lp = tl - jax.nn.logsumexp(logits, axis=-1)
        return jnp.sum(lp * probe)

    _fd_check(scalar_h, h32, [(0, 0), (7, 13), (63, 31)])


def test_grad_probe_is_not_trivially_satisfiable():
    """The cos-probe cotangent catches a 0.9x-scaled backward (the
    WRONG_GRAD cheater's bug) — sanity that the contract has teeth."""
    p = get_problem("rmsnorm")
    inputs = p.make_inputs(jax.random.PRNGKey(34), p.case_by_name("tiny"))
    ref_g = p.grad_outputs(lambda *i: p.reference(*i), *inputs)
    cal_g = p.grad_outputs(lambda *i: p.reference_bf16(*i), *inputs)
    tol = tolerance_from_reference(ref_g, cal_g)
    scaled = tuple(0.9 * g for g in ref_g)
    ok, _ = check_tolerance(error_stats(scaled, ref_g), tol)
    assert not ok
