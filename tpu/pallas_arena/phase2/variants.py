"""Hand-written RMSNorm candidate variants shared by the phase-2/4
shakedowns.

Families: honest XLA (bf16-out == the baseline's contract; an f32-out
variant kept as an honest-but-traffic-penalized case), real Pallas kernels
(the layer-2 "naive-but-correct Pallas kernel must pass" goldens at three
requested block sizes — VMEM-budgeted and row-padded after the phase-2
CompileTimeScopedVmemOom lesson), the overflow-robust unjitted golden, and
the cheater suite. Every Pallas variant ships a custom_vjp (task 0 is
fwd+bwd) with an analytic XLA backward.
"""

HONEST_XLA = """
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

# an independent second copy (different text -> different hash) for the
# same-kernel-regrade +/-3% invariant without cache interference
HONEST_XLA_B = HONEST_XLA.replace("_impl", "_impl_b")

# f32 output: honest but moves 2x the output bytes of the bf16-out baseline
# — on a memory-bound task that is a REAL ~20% slowdown (the phase-2
# "honest-xla scored 0.80" mystery, kept as a measured regression)
HONEST_XLA_F32OUT = """
import functools
import jax
import jax.numpy as jnp

@functools.partial(jax.jit)
def _impl_f32(x, g):
    x32 = x.astype(jnp.float32)
    var = jnp.mean(jnp.square(x32), axis=-1, keepdims=True)
    return x32 * jax.lax.rsqrt(var + 1e-6) * g

def kernel(x, g):
    return _impl_f32(x, g)
"""


def make_pallas_variant(block_rows: int) -> str:
    return f"""
import functools
import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl

_BR = {block_rows}
_VMEM_BUDGET = 24 * 1024 * 1024  # stay clear of the 32M scoped-vmem limit
# Scoped VMEM per (row x padded-col) element. The phase-3 formula counted
# only the bf16 in/out blocks, double-buffered (8 B/elt), and still OOM'd:
# the kernel body upcasts to f32, so x32, x*x and the f32 product are all
# live inside the block too. Phase 4 measured the truth on silicon — a
# 256-row block at dp=6144 requested 36.25 M, i.e. 23.05 B/elt — so the
# budget now comes from that measurement with a little margin.
_BYTES_PER_ELT = 24

def _make_kernel(d):
    def _kern(x_ref, g_ref, o_ref):
        x = x_ref[...].astype(jnp.float32)
        s = jnp.sum(x * x, axis=-1, keepdims=True)
        inv = jax.lax.rsqrt(s / d + 1e-6)
        o_ref[...] = (x * inv * g_ref[...].astype(jnp.float32)).astype(jnp.bfloat16)
    return _kern

@jax.jit
def _fwd(x, g):
    rows, d = x.shape
    dp = ((d + 127) // 128) * 128
    # Budget the block against the measured per-element scoped-VMEM cost
    # (phase-2 lesson: an f32-out 256-row block at d=8192 blew the 32M
    # scoped limit; phase-4 lesson: the f32 working set counts too)
    br = _BR
    while br > 16 and br * dp * _BYTES_PER_ELT > _VMEM_BUDGET:
        br //= 2
    rp = ((rows + br - 1) // br) * br
    xp = jnp.pad(x, ((0, rp - rows), (0, dp - d))) if (rp != rows or dp != d) else x
    gp = jnp.pad(g, ((0, dp - d),)) if dp != d else g
    out = pl.pallas_call(
        _make_kernel(d),
        grid=(rp // br,),
        in_specs=[pl.BlockSpec((br, dp), lambda i: (i, 0)),
                  pl.BlockSpec((1, dp), lambda i: (0, 0))],
        out_specs=pl.BlockSpec((br, dp), lambda i: (i, 0)),
        out_shape=jax.ShapeDtypeStruct((rp, dp), jnp.bfloat16),
    )(xp, gp.reshape(1, dp))
    return out[:rows, :d]

@jax.custom_vjp
def _rms(x, g):
    return _fwd(x, g)

def _rms_f(x, g):
    return _fwd(x, g), (x, g)

def _rms_b(res, ct):
    x, g = res
    ct32 = ct.astype(jnp.float32)
    x32 = x.astype(jnp.float32)
    g32 = g.astype(jnp.float32)
    d = x.shape[-1]
    var = jnp.mean(x32 * x32, axis=-1, keepdims=True)
    inv = jax.lax.rsqrt(var + 1e-6)
    t = jnp.sum(ct32 * g32 * x32, axis=-1, keepdims=True)
    gx = ct32 * g32 * inv - x32 * inv ** 3 * t / d
    gg = jnp.sum(ct32 * x32 * inv, axis=0)
    return gx.astype(x.dtype), gg.astype(g.dtype)

_rms.defvjp(_rms_f, _rms_b)

def kernel(x, g):
    return _rms(x, g)
"""


PALLAS_BR16 = make_pallas_variant(16)
PALLAS_BR256 = make_pallas_variant(256)
PALLAS_BR512 = make_pallas_variant(512)


def shakedown_variants() -> dict[str, str]:
    """name -> code for the acceptance battery (cheaters shared with the
    CPU battery library)."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from pallas_arena.tests import candidates as cand

    return {
        # must PASS
        "honest-xla": HONEST_XLA,
        "honest-xla-b": HONEST_XLA_B,
        "honest-xla-f32out": HONEST_XLA_F32OUT,
        "unjitted-robust": cand.UNJITTED_ROBUST_RMSNORM,
        "pallas-br16": PALLAS_BR16,
        "pallas-br256": PALLAS_BR256,
        "pallas-br512": PALLAS_BR512,
        # must FAIL (cheaters / bugs)
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
        "timer-tamperer": cand.TIMER_TAMPERER_RMSNORM,
    }


# legacy phase-2 (fork-per-candidate child) expectations
EXPECT_PASS = {
    "honest-xla",
    "honest-xla-b",
    "honest-xla-f32out",
    "unjitted-robust",
    "pallas-br16",
    "pallas-br256",
    "pallas-br512",
    "timer-tamperer",
}
EXPECT_SLOW = {"timer-tamperer"}  # passed but score < 1


# phase-4 persistent-worker (jax.export) expectations. Structural
# neutralizations of the export path: python-level call counting
# (split-personality) and host-RNG noise (nondeterministic) are baked away
# at trace time, so those cheats can no longer EXIST on the judge — the
# artifacts grade as their honest selves.
WORKER_EXPECT_PASS = EXPECT_PASS | {"split-personality", "nondeterministic"}
WORKER_EXPECT_SLOW = {"timer-tamperer"}
WORKER_EXPECT_FAIL_GATE = {
    "wrong-eps": ("correctness",),
    "cached-output": ("correctness",),
    "aliased-reference": ("ast",),
    "obfuscated-import": ("poison_stub", "aot_export"),
    "seed-reader": ("correctness",),
    "memoizer": ("aot_export", "correctness"),
    "wrong-grad": ("gradient",),
    "no-kernel": ("exec",),
}
