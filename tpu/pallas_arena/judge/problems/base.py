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


def _leaves(out) -> list[np.ndarray]:
    if isinstance(out, (tuple, list)):
        return [np.asarray(o, dtype=np.float64) for o in out]
    return [np.asarray(out, dtype=np.float64)]


def error_stats(cand, ref) -> dict:
    """Per-element |cand - ref| distribution stats across all output leaves,
    normalized per element by (|ref| + 1): a scale-aware absolute/relative
    hybrid so huge-magnitude outputs don't drown small-magnitude rows."""
    cl, rl = _leaves(cand), _leaves(ref)
    if len(cl) != len(rl):
        return {
            "finite": False,
            "max": float("inf"),
            "q99": float("inf"),
            "mean": float("inf"),
            "why": "output arity mismatch",
        }
    errs = []
    for c, r in zip(cl, rl):
        if c.shape != r.shape:
            return {
                "finite": False,
                "max": float("inf"),
                "q99": float("inf"),
                "mean": float("inf"),
                "why": f"shape mismatch {c.shape} vs {r.shape}",
            }
        e = np.abs(c - r) / (np.abs(r) + 1.0)
        errs.append(e.reshape(-1))
    e = np.concatenate(errs) if errs else np.zeros(1)
    finite = bool(np.isfinite(np.concatenate([c.reshape(-1) for c in cl])).all())
    return {
        "finite": finite,
        "max": float(np.max(e)) if e.size else 0.0,
        "q99": float(np.quantile(e, 0.99)) if e.size else 0.0,
        "mean": float(np.mean(e)) if e.size else 0.0,
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
        return [c for c in self.shape_cases() if not c.holdout and c.smoke == smoke]

    def holdout_cases(self, smoke: bool = False) -> list[ShapeCase]:
        return [c for c in self.shape_cases() if c.holdout and c.smoke == smoke]

    def case_by_name(self, name: str) -> ShapeCase:
        for c in self.shape_cases():
            if c.name == name:
                return c
        raise KeyError(f"{self.name}: no shape case named {name!r}")
