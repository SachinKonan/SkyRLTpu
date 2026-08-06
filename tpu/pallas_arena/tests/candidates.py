"""Candidate-code library for the battery: goldens and the cheater suite.

Every cheater here encodes one known hack class from DESIGN.md's
adversarial addendum, and every one of them MUST fail grading.
"""

HONEST_RMSNORM = """
import functools
import jax
import jax.numpy as jnp

@functools.partial(jax.jit)
def _impl(x, g):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return (x32 * jax.lax.rsqrt(var + 1e-6) * g).astype(x.dtype)

def kernel(x, g):
    return _impl(x, g)
"""

# honest, unjitted, overflow-ROBUST (pre-scaled mean): the phase-2
# "unjitted-honest" golden re-armed — eager op-by-op dispatch is a
# legitimate (slow) implementation and must pass all gates including the
# near-overflow-bf16 adversarial vector.
UNJITTED_ROBUST_RMSNORM = """
import jax
import jax.numpy as jnp

def kernel(x, g):
    x32 = x.astype(jnp.float32)
    m = jnp.maximum(jnp.max(jnp.abs(x32), axis=-1, keepdims=True), 1.0)
    var = jnp.mean(jnp.square(x32 / m), axis=-1, keepdims=True) * m * m
    return (x32 * jax.lax.rsqrt(var + 1e-6) * g).astype(x.dtype)
"""

# subtly wrong: eps inside the rsqrt is 1e-3 instead of 1e-6 (the classic
# wrong-LN-epsilon bug from DESIGN.md test layer 2)
WRONG_EPS_RMSNORM = """
import jax
import jax.numpy as jnp

def kernel(x, g):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(var + 1e-3) * g
"""

# --- cheater 1: cached-output. Precomputes the answer for the seed it
# GUESSES the judge uses (0); fresh hidden seeds make this garbage.
CACHED_OUTPUT_RMSNORM = """
import jax
import jax.numpy as jnp

_kx, _kg = jax.random.split(jax.random.PRNGKey(0))
_x = jax.random.normal(_kx, (32, 64), jnp.float32).astype(jnp.bfloat16)
_g = 1.0 + 0.1 * jax.random.normal(_kg, (64,), jnp.float32)
_x32 = _x.astype(jnp.float32)
_out = _x32 * jax.lax.rsqrt(jnp.mean(jnp.square(_x32), -1, keepdims=True) + 1e-6) * _g

def kernel(x, g):
    return _out
"""

# --- cheater 1b: SHAPE-CORRECT canned answer. Phase 4 measured that the
# cheater above never reaches the correctness gate on the persistent-worker
# path: its (32, 64) constant fails to broadcast against the real signature
# and jax.export rejects it outright. That is a fine outcome, but it left the
# "ignores its inputs and returns a plausible answer" cheat untested against
# the gate that is actually supposed to catch it. This one exports cleanly
# for every declared signature and must die on fresh hidden seeds.
CONSTANT_OUTPUT_RMSNORM = """
import jax
import jax.numpy as jnp

def kernel(x, g):
    canned = jnp.float32(0.5) * g.astype(jnp.float32)
    return jnp.broadcast_to(canned, x.shape).astype(x.dtype)
"""

# --- cheater 2: aliased/wrapped reference call (AST-rejected)
ALIASED_REFERENCE_RMSNORM = """
from pallas_arena.judge.problems.rmsnorm import PROBLEM as _p

def kernel(x, g):
    return _p.reference(x, g)
"""

# --- cheater 3: obfuscated import — invisible to AST, caught by the poison
# stubs installed in the child's sys.modules
OBFUSCATED_IMPORT_RMSNORM = """
import importlib

def kernel(x, g):
    m = importlib.import_module("pallas_" + "arena.judge.problems.rmsnorm")
    return m.PROBLEM.reference(x, g)
"""

# --- cheater 4: seed reader — scans env/argv/cwd for the grading seed; the
# seed only ever crosses on stdin (already consumed and closed), so it finds
# nothing and its "precomputed" answer is garbage.
SEED_READER_RMSNORM = """
import os
import sys
import glob
import json
import jax
import jax.numpy as jnp

_seed = None
for k, v in os.environ.items():
    if "seed" in k.lower():
        try:
            _seed = int(v)
        except ValueError:
            pass
for a in sys.argv:
    if a.isdigit():
        _seed = int(a)
for f in glob.glob("*.json") + glob.glob("/tmp/arena-grade-*/config.json"):
    try:
        _seed = json.load(open(f)).get("seed", _seed)
    except Exception:
        pass

def kernel(x, g):
    if _seed is None:
        return jnp.zeros(x.shape, jnp.float32)  # found nothing: bluff fails
    kx, kg = jax.random.split(jax.random.PRNGKey(_seed))
    xx = jax.random.normal(kx, x.shape, jnp.float32).astype(jnp.bfloat16)
    x32 = xx.astype(jnp.float32)
    return x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), -1, keepdims=True) + 1e-6) * g
"""

# --- cheater 5: timer tamperer — patches time.perf_counter to zero and then
# runs a deliberately SLOW but correct kernel, hoping the judge times it at
# zero cost. The harness holds a direct reference to the real counter, so
# the slowdown is measured honestly (score << 1).
TIMER_TAMPERER_RMSNORM = """
import time
time.perf_counter = lambda: 0.0
try:
    import builtins  # go further: shadow module-level lookups everywhere
    builtins.__dict__["perf_counter"] = lambda: 0.0
except Exception:
    pass

import jax
import jax.numpy as jnp

def kernel(x, g):
    x32 = x.astype(jnp.float32)
    out = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), -1, keepdims=True) + 1e-6) * g
    waste = x32
    # deliberately slow (unjitted op-by-op dispatch); kept short enough that
    # jax.grad's residual chain fits HBM at production shapes (phase-2: 200
    # sins x 134MB residuals OOM'd a v6e-1 during the gradient probe)
    for _ in range(24):
        waste = jnp.sin(waste)
    return (out + 0.0 * waste).astype(x.dtype)
"""

# --- cheater 6: python-level memoizer — returns its first result forever.
# Fresh inputs per correctness seed / timed iteration make the stale answer
# wrong immediately.
MEMOIZER_RMSNORM = """
import jax
import jax.numpy as jnp

_cache = {}

def kernel(x, g):
    if "out" not in _cache:
        x32 = x.astype(jnp.float32)
        _cache["out"] = x32 * jax.lax.rsqrt(
            jnp.mean(jnp.square(x32), -1, keepdims=True) + 1e-6) * g
    return _cache["out"]
"""

# --- cheater 7: split personality — correct for the first N calls (enough
# to clear the gates), then fast garbage during timing. Killed by verifying
# correctness ON OUTPUTS PRODUCED BY TIMED INVOCATIONS.
SPLIT_PERSONALITY_RMSNORM = """
import jax
import jax.numpy as jnp

_calls = [0]

def kernel(x, g):
    _calls[0] += 1
    if _calls[0] <= 14:
        x32 = x.astype(jnp.float32)
        return x32 * jax.lax.rsqrt(
            jnp.mean(jnp.square(x32), -1, keepdims=True) + 1e-6) * g
    return jnp.zeros(x.shape, jnp.float32)
"""

# --- cheater 8: wrong backward — forward is perfect, custom_vjp backward is
# scaled 0.9x. Caught by the gradient contract, not by fwd correctness.
WRONG_GRAD_RMSNORM = """
import jax
import jax.numpy as jnp

@jax.custom_vjp
def _f(x32, g):
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(var + 1e-6) * g

def _f_fwd(x32, g):
    return _f(x32, g), (x32, g)

def _f_bwd(res, cot):
    x32, g = res
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(var + 1e-6)
    gx = (cot * g * inv
          - x32 * inv**3 * jnp.mean(cot * g * x32, axis=-1, keepdims=True))
    return 0.9 * gx, jnp.sum(cot * x32 * inv, axis=0)  # 0.9x: wrong on purpose

_f.defvjp(_f_fwd, _f_bwd)

def kernel(x, g):
    return _f(x.astype(jnp.float32), g)
"""

# --- resource abusers
MEMORY_HOG = """
import numpy as np
_a = np.ones(6_000_000_000, dtype=np.uint8)  # 6 GB > RLIMIT_AS

def kernel(x, g):
    return x
"""

SLEEPER = """
import time
time.sleep(300)

def kernel(x, g):
    return x
"""

NO_KERNEL = """
def not_the_entrypoint(x, g):
    return x
"""

NONDETERMINISTIC_RMSNORM = """
import numpy as np
import jax
import jax.numpy as jnp

def kernel(x, g):
    x32 = x.astype(jnp.float32)
    out = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), -1, keepdims=True) + 1e-6) * g
    # tiny fresh noise: well inside tolerance, but flips low mantissa bits
    return out + np.random.default_rng().normal(0, 1e-6, out.shape).astype(np.float32)
"""

# honest naive splash-attention candidate (no pallas_call: rejected when the
# problem enforces pallas, passes gates when enforcement is off)
NAIVE_SPLASH = """
import jax
import jax.numpy as jnp

def kernel(q, k, v, segment_ids):
    q32, k32, v32 = (t.astype(jnp.float32) for t in (q, k, v))
    s = q.shape[1]
    idx = jnp.arange(s)
    live = segment_ids != 0
    mask = ((idx[:, None] >= idx[None, :])
            & (segment_ids[:, None] == segment_ids[None, :])
            & live[:, None] & live[None, :])
    logits = jnp.einsum("hqd,hkd->hqk", q32, k32)
    neg = -0.7 * jnp.finfo(jnp.float32).max
    logits = jnp.where(mask[None], logits, neg)
    m = jnp.max(logits, axis=-1, keepdims=True)
    p = jnp.exp(logits - m)
    p = jnp.where(mask[None], p, 0.0)
    row_live = mask.any(-1)
    p = jnp.where(row_live[None, :, None],
                  p / jnp.maximum(p.sum(-1, keepdims=True), 1e-30), 0.0)
    return jnp.einsum("hqk,hkd->hqd", p, v32)
"""
