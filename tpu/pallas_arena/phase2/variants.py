"""Hand-written RMSNorm candidate variants for the phase-2 shakedown.

Three families: honest XLA, real Pallas kernels (the layer-2 "naive-but-
correct Pallas kernel must pass" goldens, at three block sizes), and the
cheater suite re-run on silicon. Every Pallas variant ships a custom_vjp
(task 0 is fwd+bwd) with an analytic XLA backward.
"""

HONEST_XLA = """
import functools
import jax
import jax.numpy as jnp

@functools.partial(jax.jit)
def _impl(x, g):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(var + 1e-6) * g

def kernel(x, g):
    return _impl(x, g)
"""

# an independent second copy (different text -> different hash) for the
# same-kernel-regrade +/-3% invariant without cache interference
HONEST_XLA_B = HONEST_XLA.replace("_impl", "_impl_b")


def make_pallas_variant(block_rows: int) -> str:
    return f"""
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

_BR = {block_rows}

def _make_kernel(d):
    def _kern(x_ref, g_ref, o_ref):
        x = x_ref[...].astype(jnp.float32)
        s = jnp.sum(x * x, axis=-1, keepdims=True)
        inv = jax.lax.rsqrt(s / d + 1e-6)
        o_ref[...] = x * inv * g_ref[...].astype(jnp.float32)
    return _kern

@jax.jit
def _fwd(x, g):
    rows, d = x.shape
    dp = ((d + 127) // 128) * 128
    br = _BR if rows % _BR == 0 else 16  # 16 = bf16 sublane tile
    xp = jnp.pad(x, ((0, 0), (0, dp - d))) if dp != d else x
    gp = jnp.pad(g, ((0, dp - d),)) if dp != d else g
    out = pl.pallas_call(
        _make_kernel(d),
        grid=(rows // br,),
        in_specs=[pl.BlockSpec((br, dp), lambda i: (i, 0)),
                  pl.BlockSpec((1, dp), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((br, dp), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((rows, dp), jnp.float32),
    )(xp, gp.reshape(1, dp))
    return out[:, :d]

@jax.custom_vjp
def _rms(x, g):
    return _fwd(x, g)

def _rms_f(x, g):
    return _fwd(x, g), (x, g)

def _rms_b(res, ct):
    x, g = res
    x32 = x.astype(jnp.float32)
    g32 = g.astype(jnp.float32)
    d = x.shape[-1]
    var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(var + 1e-6)
    t = jnp.sum(ct * g32 * x32, axis=-1, keepdims=True)
    gx = ct * g32 * inv - x32 * inv ** 3 * t / d
    gg = jnp.sum(ct * x32 * inv, axis=0)
    return gx.astype(x.dtype), gg.astype(g.dtype)

_rms.defvjp(_rms_f, _rms_b)

def kernel(x, g):
    return _rms(x, g)
"""


PALLAS_BR16 = make_pallas_variant(16)
PALLAS_BR256 = make_pallas_variant(256)
PALLAS_BR512 = make_pallas_variant(512)

UNJITTED_HONEST = """
import jax
import jax.numpy as jnp

def kernel(x, g):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    out = x32 * jax.lax.rsqrt(var + 1e-6) * g
    for _ in range(6):  # deliberately slow: honest slowdown must score < 1
        out = out + 0.0 * jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return out
"""


def shakedown_variants() -> dict[str, str]:
    """name -> code for the ~20-variant acceptance run (cheaters included
    from the shared battery library)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pallas_arena.tests import candidates as cand

    return {
        # must PASS
        "honest-xla": HONEST_XLA,
        "honest-xla-b": HONEST_XLA_B,
        "pallas-br16": PALLAS_BR16,
        "pallas-br256": PALLAS_BR256,
        "pallas-br512": PALLAS_BR512,
        # must PASS with score < 1 (honest slowdown)
        "unjitted-honest": UNJITTED_HONEST,
        # must FAIL (cheaters / bugs, silicon re-run of the CPU battery)
        "wrong-eps": cand.WRONG_EPS_RMSNORM,
        "cached-output": cand.CACHED_OUTPUT_RMSNORM,
        "aliased-reference": cand.ALIASED_REFERENCE_RMSNORM,
        "obfuscated-import": cand.OBFUSCATED_IMPORT_RMSNORM,
        "seed-reader": cand.SEED_READER_RMSNORM,
        "memoizer": cand.MEMOIZER_RMSNORM,
        "split-personality": cand.SPLIT_PERSONALITY_RMSNORM,
        "wrong-grad": cand.WRONG_GRAD_RMSNORM,
        "nondeterministic": cand.NONDETERMINISTIC_RMSNORM,
        "no-kernel": cand.NO_KERNEL,
        # must NOT look fast (tamper attempt measured honestly)
        "timer-tamperer": cand.TIMER_TAMPERER_RMSNORM,
    }


EXPECT_PASS = {
    "honest-xla",
    "honest-xla-b",
    "pallas-br16",
    "pallas-br256",
    "pallas-br512",
    "unjitted-honest",
    "timer-tamperer",
}
EXPECT_SLOW = {"unjitted-honest", "timer-tamperer"}  # passed but score < 1
