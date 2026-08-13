"""Problem contract for the arena + calibrated tolerances and error tails.

Every task module defines one ``Problem`` subclass instance. The contract
pins, per task:
  * the candidate kernel signature (``kernel(*inputs)``),
  * the declared shape set + at least one HOLDOUT shape (logged, unscored),
  * an fp32 closed-form reference and the production baseline-to-beat,
  * the adversarial vector library (tolerance-exploitation defense),
  * the fwd-only vs fwd+bwd contract (``has_bwd``),
  * banned imports/calls and whether a real ``pallas_call`` is required.

Tolerance is CALIBRATED, never fixed: the reference run at bf16 precision
defines the error scale, and a candidate must stay within
``TOL_MULTIPLier×`` of that scale on BOTH the max and the tail quantile of
the per-element error distribution (a truncated-softmax approximator can
pass a global allclose on Gaussian inputs forever; it cannot pass the tail
check on the adversarial vectors).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

TOL_MULTIPLIER = 1.5  # candidate error budget vs reference's own bf16 error
ABS_FLOOR = 1e-6  # absolute floor so exact-zero reference error
# does not demand bitwise equality


@dataclass(frozen=True)
class ShapeCase:
    """One entry of a task's shape set."""

    name: str
    dims: dict[str, Any]
    holdout: bool = False  # logged-unscored (declared-set overfit detector)
    smoke: bool = False  # tiny CPU-battery case, never scored in prod
    # PROBE cases exist so a task can be graded end-to-end on ONE chip. The
    # production shape sets are deliberately full-size, and for two tasks the
    # fp32 REFERENCE cannot run there at all on a 32 GB judge: splash's
    # closed form materializes [heads, seq, seq] (10.9 GB at h8-s18432,
    # 43.5 GB at h32) and FLCE's materializes [n, v] fp32 (44.8 GB at
    # 73728x151936). Probe cases are the same op at a size one chip can hold,
    # selected explicitly by name (`--cases`), and are NEVER part of the
    # default scored/holdout sets, so nothing about a production run changes.
    probe: bool = False


@dataclass(frozen=True)
class AdversarialCase:
    """One adversarial structured-input vector.

    ``make_inputs(key)`` builds the inputs; ``expect(ref_out, inputs)`` may
    assert extra structural facts about the REFERENCE output (e.g. a fully
    masked row must be exactly 0, not NaN).
    """

    name: str
    make_inputs: Callable[[Any], tuple]
    expect: Callable[[Any, tuple], None] | None = None


# q99 is estimated from a deterministic strided subsample once a leaf gets
# big. Phase 4 measured why: the old host-side path upcast every leaf to
# float64, pulled it off the chip and built ~5 full-size temporaries per
# call — at 32768x8192 that is ~10 GB of host traffic, and the worker makes
# ~90 of these calls per candidate (tolerance calibration runs 4 honest
# implementations per fixture). The host-side np.quantile over ~14 G
# elements per candidate was the single largest term in the 261 s attempt-1
# warm cost. Now the elementwise error, max, mean and finiteness are EXACT
# on device and only a capped sample crosses to the host for the tail
# quantile. Striding is deterministic, so regrades stay bit-stable; and it
# is unexploitable — a candidate never learns the hidden fixture seed, the
# reference output, or which elements are compared, and `max` remains exact
# over every element.
Q99_SAMPLE_CAP = 1 << 19  # 524,288 elements: q99 sampling error ~0.4% at the
# tail, i.e. two orders below the 1.5x calibrated margin it feeds


def _leaves(out) -> list:
    return list(out) if isinstance(out, (tuple, list)) else [out]


def _bad(why: str) -> dict:
    return {"finite": False, "max": float("inf"), "q99": float("inf"), "mean": float("inf"), "why": why}


def _sample_stride(n: int, row_len: int) -> int:
    """Stride for the q99 subsample, chosen COPRIME to the row length.

    A plain ceil(n / cap) stride is usually a power of two, and so are the
    row lengths (4096, 8192, ...) — the sample would then only ever land on
    a handful of column indices, repeated in every row. Error in a tiled
    kernel is column-correlated (edge tiles, padding), and the strides are
    derivable from the public shapes, so that would be a tail-check blind
    spot a candidate could aim at. With gcd(stride, row_len) = 1 the walk
    visits every column index in turn. (`max` is exact either way, so this
    only hardens the q99 tail.)"""
    import math

    s = max(1, -(-n // Q99_SAMPLE_CAP))  # ceil-div: sample <= cap
    while s < n and math.gcd(s, row_len) != 1:
        s += 1
    return s


_LEAF_STATS = None


def _leaf_stats_fn():
    """One fused pass per leaf: XLA never materializes the error array, so
    peak memory is unchanged and only (max, sum, finite, sample) come back.
    The upcasts live INSIDE the jit so no full-size f32 temporary is
    materialized eagerly on the way in."""
    global _LEAF_STATS
    if _LEAF_STATS is None:
        import functools

        import jax
        import jax.numpy as jnp

        @functools.partial(jax.jit, static_argnums=2)
        def _f(c, r, stride):
            c32 = c.astype(jnp.float32)
            r32 = r.astype(jnp.float32)
            e = (jnp.abs(c32 - r32) / (jnp.abs(r32) + 1.0)).reshape(-1)
            return jnp.max(e), jnp.sum(e), e[::stride], jnp.isfinite(c32).all()

        _LEAF_STATS = _f
    return _LEAF_STATS


def error_stats(cand, ref) -> dict:
    """Per-element |cand - ref| distribution stats across all output leaves,
    normalized per element by (|ref| + 1): a scale-aware absolute/relative
    hybrid so huge-magnitude outputs don't drown small-magnitude rows.

    Computed on whatever device the arrays already live on (float32); only
    scalars and the capped q99 sample are transferred."""
    import jax.numpy as jnp

    cl, rl = _leaves(cand), _leaves(ref)
    if len(cl) != len(rl):
        return _bad("output arity mismatch")

    leaf_stats = _leaf_stats_fn()
    maxes, total, counts, finite, samples = [], 0.0, 0, True, []
    for c, r in zip(cl, rl):
        ca, ra = jnp.asarray(c), jnp.asarray(r)
        if ca.shape != ra.shape:
            return _bad(f"shape mismatch {ca.shape} vs {ra.shape}")
        n = int(ca.size)
        if n == 0:
            continue
        stride = _sample_stride(n, ca.shape[-1] if ca.ndim else 1)
        mx, sm, sample, fin = leaf_stats(ca, ra, stride)
        maxes.append(float(mx))
        total += float(sm)
        counts += n
        finite = finite and bool(fin)
        samples.append(np.asarray(sample))

    if not counts:
        return {"finite": finite, "max": 0.0, "q99": 0.0, "mean": 0.0}
    sample = np.concatenate(samples) if len(samples) > 1 else samples[0]
    return {
        "finite": finite,
        "max": max(maxes),
        "q99": float(np.quantile(sample, 0.99)),
        "mean": total / counts,
    }


def tolerance_from_reference(ref_fp32, ref_bf16) -> dict:
    """Calibrated tolerance: TOL_MULTIPLIER × the reference's OWN bf16 error
    (max and q99 of the same per-element error metric), floored."""
    s = error_stats(ref_bf16, ref_fp32)
    return {
        "max": TOL_MULTIPLIER * max(s["max"], ABS_FLOOR),
        "q99": TOL_MULTIPLIER * max(s["q99"], ABS_FLOOR),
    }


def check_tolerance(stats: dict, tol: dict) -> tuple[bool, str]:
    if not stats.get("finite", False):
        return False, f"non-finite or malformed output ({stats.get('why', 'NaN/inf')})"
    if stats["max"] > tol["max"]:
        return False, (f"per-element max error {stats['max']:.3e} exceeds " f"calibrated tolerance {tol['max']:.3e}")
    if stats["q99"] > tol["q99"]:
        return False, (
            f"per-element q99 error tail {stats['q99']:.3e} exceeds " f"calibrated tolerance {tol['q99']:.3e}"
        )
    return True, "ok"


class BaselineUnavailable(RuntimeError):
    """Raised by ``baseline()`` when the production kernel can't run here
    (e.g. a TPU Pallas kernel on a CPU test host)."""


class Problem(abc.ABC):
    """Base class for arena tasks. Subclasses are stateless: all methods are
    pure functions of (key, case) so fresh hidden seeds fully determine the
    inputs — a candidate can never precompute them."""

    name: str = "?"
    version: str = "1"
    has_bwd: bool = False
    require_pallas: bool = False
    memory_bound: bool = False
    banned_import_prefixes: tuple[str, ...] = ()
    banned_call_names: tuple[str, ...] = ()
    kernel_entrypoint: str = "kernel"
    # Which shape case the adversarial vector library is built on. Instance-
    # settable so a judge grading a non-default case set does not silently
    # require the candidate to also trace at the CPU-battery `tiny` shapes,
    # which no prompt ever declares.
    adversarial_case_name: str = "tiny"

    # Blanket bans for every task: the judge's own package, the whole
    # jax-bundled pallas ops tree (candidates write kernels, not wrappers),
    # and every external baseline source.
    UNIVERSAL_BANS: tuple[str, ...] = (
        "pallas_arena",
        "tpu.pallas_arena",
        "jax.experimental.pallas.ops",
        "tpu_inference",
        "recurrentgemma",
        "MaxText",
        "maxtext",
        "skyrl",
    )

    @property
    def all_banned_prefixes(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(self.UNIVERSAL_BANS + self.banned_import_prefixes))

    # ------------------------------------------------------------------ data
    @abc.abstractmethod
    def shape_cases(self) -> list[ShapeCase]: ...

    @abc.abstractmethod
    def make_inputs(self, key, case: ShapeCase) -> tuple:
        """Generate inputs ON DEVICE from a jax PRNG key derived from the
        hidden seed. Never reads globals/env."""

    # ------------------------------------------------------- reference/baseline
    @abc.abstractmethod
    def reference(self, *inputs):
        """fp32 closed-form reference (correctness oracle)."""

    def reference_bf16(self, *inputs):
        """Same closed form at bf16 working precision (tolerance calibration).
        Default: cast float inputs to bf16, run reference, and round the
        OUTPUTS through bf16 too — a candidate producing bf16 results (the
        normal TPU kernel contract) must not be held to fp32 bitness."""
        import jax.numpy as jnp

        cast = tuple(
            x.astype(jnp.bfloat16) if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating) else x
            for x in inputs
        )
        out = self.reference(*cast)

        def _round(o):
            return o.astype(jnp.bfloat16).astype(jnp.float32)

        if isinstance(out, (tuple, list)):
            return tuple(_round(o) for o in out)
        return _round(out)

    @abc.abstractmethod
    def baseline(self, *inputs):
        """The production baseline-to-beat. May raise BaselineUnavailable."""

    def baseline_available(self) -> tuple[bool, str]:
        try:
            import jax

            case = next(c for c in self.shape_cases() if c.smoke)
            key = jax.random.PRNGKey(0)
            self.baseline(*self.make_inputs(key, case))
            return True, "ok"
        except BaselineUnavailable as e:
            return False, str(e)
        except StopIteration:
            return False, "no smoke case to probe with"

    # ------------------------------------------------------------- calibration
    def honest_variants(self) -> list[Callable]:
        """Legitimate alternative implementations spanning the honest
        precision/reduction space (different output dtypes, reduction
        orders, scaling paths). Tolerance calibration takes the WORST
        spread across `reference_bf16` + these, so an honest kernel with a
        different-but-valid numeric path is never rejected (phase-2
        shakedown lesson: three honest goldens failed the
        reference-bf16-only 1.5x margin). Default: none (reference_bf16
        alone — the phase-1 behavior)."""
        return []

    # When True, the production baseline participates in tolerance calibration.
    # Principle: a candidate must never be held to a TIGHTER standard than the
    # production kernel itself. Set it ONLY where the baseline solves the exact
    # contract -- measured necessity on ragged_paged_attention (job 3692058:
    # Google's own kernel misses the reference_bf16-calibrated band at 1.05x),
    # and measured harm on splash, whose kernel differs on the padding contract
    # and would widen the band with a CONTRACT error rather than a numeric one.
    baseline_calibrates: bool = False

    def calibrated_tolerance(self, inputs, ref32) -> dict:
        """Per-input tolerance: TOL_MULTIPLIER x the max error (max and q99
        tails) of ANY honest implementation vs the fp32 reference."""
        stats = [error_stats(self.reference_bf16(*inputs), ref32)]
        for variant in self.honest_variants():
            s = error_stats(variant(*inputs), ref32)
            if s.get("finite"):
                # a variant that goes non-finite on this input is NOT honest
                # here; it must never widen the tolerance to infinity
                stats.append(s)
        if self.baseline_calibrates:
            try:
                s = error_stats(self.baseline(*inputs), ref32)
                if s.get("finite"):
                    stats.append(s)
            except Exception:  # noqa: BLE001 -- baseline unavailable: calibrate without it
                pass
        return {
            "max": TOL_MULTIPLIER * max(max(s["max"] for s in stats), ABS_FLOOR),
            "q99": TOL_MULTIPLIER * max(max(s["q99"] for s in stats), ABS_FLOOR),
        }

    # -------------------------------------------------------------- gradients
    def grad_outputs(self, kernel_fn: Callable, *inputs):
        """For has_bwd tasks: return the gradient pytree of a fixed scalar
        functional of the kernel output wrt the differentiable inputs.
        Subclasses with has_bwd must override."""
        raise NotImplementedError

    # ------------------------------------------------------------ adversarial
    def adversarial_cases(self) -> list[AdversarialCase]:
        return []

    # ----------------------------------------------------------------- memory
    def bytes_moved(self, case: ShapeCase) -> int | None:
        """Minimum HBM traffic for the op (speed-of-light denominator)."""
        return None

    # ------------------------------------------------------------------ misc
    def abstract_inputs(self, case: ShapeCase):
        """ShapeDtypeStructs for the AOT pre-gate (no real data, no device)."""
        import jax

        return jax.eval_shape(
            lambda k: self.make_inputs(k, case),
            jax.ShapeDtypeStruct((2,), np.dtype("uint32")),
        )

    def scored_cases(self, smoke: bool = False) -> list[ShapeCase]:
        return [c for c in self.shape_cases() if not c.holdout and c.smoke == smoke and not c.probe]

    def holdout_cases(self, smoke: bool = False) -> list[ShapeCase]:
        return [c for c in self.shape_cases() if c.holdout and c.smoke == smoke and not c.probe]

    def probe_cases(self) -> list[ShapeCase]:
        return [c for c in self.shape_cases() if c.probe]

    def case_by_name(self, name: str) -> ShapeCase:
        for c in self.shape_cases():
            if c.name == name:
                return c
        raise KeyError(f"{self.name}: no shape case named {name!r}")
