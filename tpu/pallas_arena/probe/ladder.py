"""The P4 PRIMITIVES prelude: tested helpers the model composes.

Rung P4 of the prompt ladder hands the model a small library of *already
correct* building blocks instead of more prose. The prelude is **prepended**
to the model's own program before it is submitted, so:

  * the model never has to type the constructs that the measured failure
    histogram says it gets wrong (`dot_general`'s pair-of-pairs
    `dimension_numbers`, a shape-correct store into a Ref, the online-softmax
    rescale, the associative-scan combine signature), and
  * the prelude is **purely additive**: nothing in it is required, nothing it
    defines is a `pallas_call`, and a model that ignores it entirely produces
    exactly the program it would have produced at rung P3. That is what makes
    the same control program valid for both rungs.

Every helper leaves the decisions that move the SCORE with the model -- block
and tile shapes, which blocks to skip, one-pass vs two-pass, chunk length, and
(deliberately, via `hi=`) the matmul precision/speed trade that the seam run
measured as an 18% tolerance miss. A helper that decided those would rebuild
the `tailored` trap: 16/16 passing with a reward spread below the noise floor.

`ladder_control.py` exports a program built on these helpers at every declared
probe shape, so "the primitives work" is a measurement, not a claim.
"""

from __future__ import annotations

# --------------------------------------------------------------------------
# The individual helpers, as source text. Kept as separate strings so each
# task's prelude carries ONLY what its prompt documents -- an undocumented
# helper is dead weight in a 16k context.

_HEADER = '''# ============ ARENA PRIMITIVES (provided; already defined above your code) ==
import jax as _pjax
import jax.numpy as _pjnp
'''

_DOT_F32 = '''

def dot_f32(x, y, cx, cy, hi=False):
    """Contract axis `cx` of x with axis `cy` of y, accumulating in float32.

    Wraps lax.dot_general so its dimension_numbers (a PAIR OF PAIRS) cannot be
    written wrong. `hi=True` selects Precision.HIGHEST: ~6x the passes and the
    only way to hit a tight tolerance on f32 operands. That trade is yours.
    """
    return _pjax.lax.dot_general(
        x.astype(_pjnp.float32), y.astype(_pjnp.float32),
        (((cx,), (cy,)), ((), ())),
        precision=(_pjax.lax.Precision.HIGHEST if hi else None),
        preferred_element_type=_pjnp.float32,
    )
'''

_IOTA2 = '''

def iota2(shape, dim):
    """int32 position indices along `dim`, broadcast over `shape`."""
    return _pjax.lax.broadcasted_iota(_pjnp.int32, shape, dim)
'''

_FILL_REF = '''

def fill_ref(ref, value):
    """Store a scalar into every element of a Ref, at the Ref's own shape."""
    ref[...] = _pjnp.full(ref.shape, value, ref.dtype)
'''

_ONLINE_SOFTMAX = '''

NEG = -1e30


def online_softmax(m, l, o, s, v, live, hi=False):
    """One streaming (flash) softmax update over a block of keys.

      m    [R, 1]  running row max      (initialise to NEG)
      l    [R, 1]  running row sum      (initialise to 0)
      o    [R, D]  running UNNORMALISED output (initialise to 0)
      s    [R, K]  this block's logits
      v    [K, D]  this block's values
      live [R, K] or [1, K] bool: True where the key is visible

    Returns (m, l, o) with the same shapes. Normalise ONCE at the end:
        out = jnp.where(l > 0, o / jnp.maximum(l, 1e-30), 0.0)
    which yields exactly 0 for a fully-masked row rather than NaN.
    """
    s = _pjnp.where(live, s, NEG)
    m_new = _pjnp.maximum(m, _pjnp.max(s, axis=1, keepdims=True))
    corr = _pjnp.exp(m - m_new)
    p = _pjnp.where(live, _pjnp.exp(s - m_new), 0.0)
    l_new = corr * l + _pjnp.sum(p, axis=1, keepdims=True)
    o_new = corr * o + dot_f32(p, v, 1, 0, hi=hi)
    return m_new, l_new, o_new
'''

_CHUNK_SCAN = '''

def chunk_scan(a, g):
    """h[:, t] = a[:, t] * h[:, t-1] + g[:, t], over axis 1, h[:, -1] = 0 before.

    a, g: [B, T, D] float32.  Returns h: [B, T, D] in the SAME layout (time on
    axis 1) -- no moveaxis needed afterwards. Uses lax.associative_scan, whose
    combine takes two PYTREES, not four arrays.
    """
    def _combine(left, right):
        a_l, g_l = left
        a_r, g_r = right
        return a_l * a_r, g_l * a_r + g_r

    _, h = _pjax.lax.associative_scan(_combine, (a, g), axis=1)
    return h
'''

_FOOTER = '''
# ============ END ARENA PRIMITIVES ==========================================
'''


def _prelude(*parts: str) -> str:
    return _HEADER + "".join(parts) + _FOOTER


# Per task, ONLY the helpers that task's P4 prompt documents.
PRELUDES: dict[str, str] = {
    "splash_attention": _prelude(_DOT_F32, _IOTA2, _ONLINE_SOFTMAX),
    "ragged_paged_attention": _prelude(_DOT_F32, _IOTA2, _FILL_REF, _ONLINE_SOFTMAX),
    "megablox_gmm": _prelude(_DOT_F32, _IOTA2),
    "rg_lru": _prelude(_CHUNK_SCAN),
    # FLCE's measured failure is the MATH of the recompute backward, not the
    # dialect, so its P4 rung is a stated backward CONTRACT (see
    # prompt_ladder.FLCE_BACKWARD_CONTRACT) and it ships no prelude at all.
    "flce": "",
}

# Which ladder rungs get a prelude prepended.
PRELUDE_VARIANTS = ("p4",)


def compose_ladder(task: str, variant: str, code: str) -> str:
    """Model program -> the program the judge actually grades.

    Prepended, never appended: the model's own definitions come LAST, so a
    model that redefines `dot_f32` (or defines `kernel` twice) still wins, and
    the prelude can only ever add names, never take one away.
    """
    if variant not in PRELUDE_VARIANTS:
        return code
    pre = PRELUDES.get(task) or ""
    if not pre or not code.strip():
        return code
    return pre.rstrip() + "\n\n" + code.lstrip()


def prelude_of(task: str) -> str:
    return (PRELUDES.get(task) or "").strip()
