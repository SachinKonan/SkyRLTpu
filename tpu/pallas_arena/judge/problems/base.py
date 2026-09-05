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
    # TENSOR-PARALLEL case: the DECLARED mesh width (0 = not a TP case).
    # Graded on a real device mesh under shard_map with the elected baseline
    # sharded identically -- a kernel that only works unsharded on one chip is
    # not a candidate for upstreaming, so sharded behaviour belongs IN the
    # reward rather than after it.
    #
    # DECLARED, not discovered, and that is load-bearing: build_signatures is
    # module-level precisely so the CPU pre-gate exports the exact signature
    # set the judge will demand. The pre-gate sees 1 device and the judge sees
    # 8, so a width derived from available devices would make them disagree and
    # the pre-gate would stop being a pre-gate. A judge with fewer devices than
    # the declared width must SKIP the case and say so, never silently grade it
    # at a different width.
    tp: int = 0
    # PROBE cases exist so a task can be graded end-to-end on ONE chip. The
    # production shape sets are deliberately full-size, and for two tasks the
    # fp32 REFERENCE cannot run there at all on a 32 GB judge: splash's
    # closed form materializes [heads, seq, seq] (10.9 GB at h8-s18432,
    # 43.5 GB at h32) and FLCE's materializes [n, v] fp32 (44.8 GB at
    # 73728x151936). Probe cases are the same op at a size one chip can hold,
    # selected explicitly by name (`--cases`), and are NEVER part of the
    # default scored/holdout sets, so nothing about a production run changes.
    probe: bool = False
    # STATIC feature knobs bound into the kernel for this case (sliding-window
    # size, logit soft-cap, ...). Empty = the plain op.
    #
    # STATIC, not traced, and that is the whole point. A sliding window passed
    # as a traced value cannot be used to SKIP blocks -- the kernel would still
    # visit every KV block and merely mask it -- which throws away the only
    # reason sliding-window attention is fast. Binding it at export time is
    # what lets a candidate specialize, exactly as the production kernels do:
    # splash takes a `LocalMask` and an `attn_logits_soft_cap` float, both
    # resolved before compilation.
    #
    # Consequence: each distinct feature combination is its own exported
    # artifact, so features multiply signatures rather than widening one.
    # BLIND: measured and reported, NEVER scored -- in either reward mode.
    #
    # This is the validation set. `general_mode` deliberately scores the holdout
    # so a kernel cannot hardcode the shapes it was shown, which is right, but
    # it leaves nothing the model is not optimizing -- so nothing that can tell
    # us whether a high reward means "learned the op" or "learned our cases".
    # A blind case answers that, and only stays able to answer it by paying
    # nothing.
    blind: bool = False
    features: "tuple[tuple[str, Any], ...]" = ()

    @property
    def feature_kwargs(self) -> dict:
        """Static kwargs to bind into kernel/reference/baseline for this case.

        A tuple-of-pairs is stored rather than a dict because ShapeCase is a
        frozen (hashable) dataclass and a dict field would break that.
        """
        return dict(self.features)


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


def grad_leaf_tolerances(ref_fp32, *cal_grads) -> list[dict]:
    """PER-LEAF calibrated tolerances for a gradient pytree.

    One pooled band over all leaves lets the noisiest gradient set the bar for
    every other one: measured on attention, dq's honest bf16 error is roughly
    an order of magnitude above dv's (tokamax's own splash test encodes the
    same fact -- dq atol 1.5 vs dv 0.15), so under a pooled band a candidate
    with a WRONG dv but a quiet dq passes. Calibrating each leaf against its
    own honest error makes each gradient answer for itself.

    Takes MULTIPLE calibration gradients and bands each leaf at the max across
    them, because reference_bf16 alone is not enough -- the v5p validation run
    (job 3719578) proved it: reference_bf16's gradient came out numerically
    equal to the fp32 reference's, the band collapsed to the 1.5e-6 absolute
    floor, and an HONEST blocked implementation's autodiff backward (error
    2.6e-3, ordinary bf16 scale from a different reduction order in reverse
    mode) was rejected. That is the exact phase-2 forward-calibration mistake
    -- the band must span what honest implementations DO, so the honest
    variants' gradients belong in it, just as their forwards belong in the
    forward band."""
    ref_leaves = _leaves(ref_fp32)
    cal_leaf_sets = [_leaves(g) for g in cal_grads if g is not None]
    tols = []
    for i, r in enumerate(ref_leaves):
        per_cal = []
        for cal in cal_leaf_sets:
            s = error_stats(cal[i], r)
            if s.get("finite"):
                per_cal.append(s)
        mx = max((s["max"] for s in per_cal), default=0.0)
        q99 = max((s["q99"] for s in per_cal), default=0.0)
        tols.append({
            "max": TOL_MULTIPLIER * max(mx, ABS_FLOOR),
            "q99": TOL_MULTIPLIER * max(q99, ABS_FLOOR),
        })
    return tols


def check_grad_tolerance(cand_g, ref_g, tols: list[dict]) -> tuple[bool, str]:
    """Leaf-wise gradient check; names the failing leaf (grad[i] = d/d inputs[i])."""
    cl, rl = _leaves(cand_g), _leaves(ref_g)
    if len(cl) != len(rl) or len(cl) != len(tols):
        return False, f"gradient arity mismatch: cand {len(cl)} ref {len(rl)} tols {len(tols)}"
    for i, (c, r, t) in enumerate(zip(cl, rl, tols)):
        ok, why = check_tolerance(error_stats(c, r), t)
        if not ok:
            return False, f"grad[{i}] (d/d input {i}): {why}"
    return True, "ok"


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
    # Does a failed backward ZERO the candidate, or just cost it the backward
    # component of the reward?
    #
    # MEASURED (job 3707571, and reproduced on CPU): JAX's generic
    # `_pallas_call_jvp_rule` re-traces the kernel body outside a grid context,
    # so `pl.program_id` trips `assert env.grid_context is not None`. Any
    # grid-based Pallas kernel is therefore NOT differentiable by generic
    # autodiff -- the only way through is `jax.custom_vjp` plus a hand-written
    # backward kernel, which is exactly what Google's splash does
    # (`_splash_attention_bwd`, 2 custom_vjp registrations).
    #
    # That bar is the real upstreaming requirement, but as a hard gate it is
    # indiscriminate: it zeroed 8/8 splash winners (program_id, no custom_vjp)
    # while rg_lru's winners passed untouched (no program_id, so generic
    # autodiff applies). One flag, two very different tasks -- and a reward
    # surface where a correct 2x-faster kernel scores the same as one that does
    # not compile carries NO learning signal at all.
    #
    # So the default is to SCORE it, not gate on it: the backward result is
    # recorded per candidate either way, which means the hard-gate verdict
    # stays fully derivable and this flips back by setting True.
    bwd_gates: bool = False
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

    def for_case(self, case):
        """A view of this problem with the case's STATIC features bound into
        every implementation at once.

        Bound as a family on purpose. The dangerous failure mode is PARTIAL
        binding inside ``calibrated_tolerance``: if the reference is windowed
        but the honest variants are not, the band widens to the difference
        between windowed and full attention -- enormous, silent, and it would
        admit almost any wrong kernel. Binding reference, baseline, variants
        and candidates together makes that state unreachable rather than
        merely avoided by discipline.

        ``reference_bf16`` needs no binding: it delegates to ``self.reference``
        and so follows the view automatically.

        Returns self when the case has no features, so the common path keeps
        its exact identity and cost.
        """
        feats = case.feature_kwargs if case is not None else {}
        if not feats:
            return self
        import copy as _copy
        import functools as _ft

        view = _copy.copy(self)
        view.reference = _ft.partial(self.reference, **feats)
        view.baseline = _ft.partial(self.baseline, **feats)
        view.honest_variants = lambda: [
            _ft.partial(v, **feats) for v in self.honest_variants()
        ]
        view.baseline_candidates = lambda: {
            k: _ft.partial(f, **feats) for k, f in self.baseline_candidates().items()
        }
        return view

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

    # GENERAL_OPTIMIZATION mode. Two consequences, both about making a reward
    # above 1.0 MEAN something beyond the shape it was measured at:
    #   * final_reward scores the holdout too (see timing.final_reward), so a
    #     kernel cannot hardcode the shapes it was shown;
    #   * the denominator is the FASTEST honest implementation at each shape
    #     (see baseline_candidates), not one named kernel -- otherwise a win can
    #     come from a mistuned denominator, which is exactly what megablox
    #     (38x untuned) and splash (10x untuned) did to every historical score.
    general_mode: bool = False

    @abc.abstractmethod
    def baseline(self, *inputs):
        """The production baseline-to-beat. May raise BaselineUnavailable."""

    def tp_specs(self, case=None):
        """PartitionSpecs for tensor-parallel grading: (in_specs, out_spec).

        Returns None when the task declares no TP axis. Each task shards along
        the axis it is ACTUALLY sharded on in production, chosen so no
        collective is needed inside the kernel -- the candidate sees a
        per-device shard and writes an ordinary kernel for it, which is exactly
        how splash (`head_shards`) and megablox (`group_offset`) are used for
        real. What TP grading then measures is whether the kernel still works
        at per-shard shapes and whether it SCALES, not whether the model can
        write collectives.
        """
        return None

    def baseline_candidates(self) -> dict[str, Callable]:
        """Named honest implementations of this task, for GENERAL mode's
        best-known denominator. The judge times every one of them at boot,
        per shape, and grades against the FASTEST -- so "beat the baseline"
        means "beat the best implementation we know of at this shape", not
        "beat the one we happened to name".

        Default: just ``baseline`` (the production kernel, whatever it binds).
        Tasks with a competitive alternative override -- e.g. megablox at probe
        shapes, where XLA `ragged_dot` is 38x faster than the Pallas kernel we
        would otherwise have scored against.
        """
        return {"production": self.baseline}

    def elected_candidates(self) -> dict:
        """baseline_candidates(), filtered by the ARENA_BASELINE env knob.

        ARENA_BASELINE=xla drops the tuned production kernel from the
        denominator, leaving the naive/XLA implementations (splash: xla-*;
        rg_lru: lax-associative-scan). Rationale (2026-09-02): against the
        production kernel every candidate scores <= 1.0 and piles up below the
        seed, so the reward has a VALIDITY gradient but no SPEED gradient --
        nothing pulls a working kernel toward faster. Against XLA, which the
        seed already beats, rewards spread ABOVE 1.0 and the slope keeps going.
        Default 'all' preserves the historical fastest-of-all election, so
        omitting the knob changes nothing.
        """
        import os
        mode = os.environ.get("ARENA_BASELINE", "all").lower()
        cands = self.baseline_candidates()
        if mode == "xla":
            kept = {k: v for k, v in cands.items()
                    if k != "production" and not k.startswith("pallas")}
            if kept:
                return kept
            # No non-production candidate for this task: fall back rather than
            # score against an empty denominator.
        return cands

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
    def grad_calibration_variants(self) -> list[Callable]:
        """Implementations whose AUTODIFF backwards calibrate the gradient
        band. Defaults to ``honest_variants()`` -- but the two bands answer
        different questions, so a task may override: megablox's forward band
        is correctly reference_bf16-only (a GMM is one fp32-accumulated
        reduction; measured, job 3689440), while its BACKWARD accumulates
        bf16-cast gradients over m rows across differently-compiled programs
        and drifts at ~3e-3 (v5p run 3722139: an fp32 ragged_dot candidate's
        own gradient failed a floor-collapsed 1.5e-6 band). Forward
        calibration stays untouched; the backward gets an honest source."""
        return self.honest_variants()

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
            try:
                s = error_stats(variant(*inputs), ref32)
            except Exception:  # noqa: BLE001
                # A variant that cannot EXPRESS this shape simply does not
                # constrain the band -- e.g. the square-MHA honest variants
                # against deepseek2's d_v != d. Skipping keeps the tolerance
                # defined by whatever honest implementations do apply, rather
                # than failing calibration for the whole case.
                continue
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
    def flops(self, case: ShapeCase) -> int | None:
        """Analytic FLOP count for one case (compute-bound problems).

        Powers the MXU-utilization line in the observation, the compute-side
        twin of bytes_moved's speed-of-light fraction. None = not modelled.
        """
        return None

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

    def tp_declared_width(self, case: ShapeCase) -> int:
        """The case's declared mesh width, validated against its own shapes.

        Returns 0 when the case is not TP or the task declares no TP axis.
        Raises if a declared width does not divide every sharded axis -- that
        is a task-definition bug and must fail loudly at import/boot rather
        than as an IndivisibleError in the middle of a grade (the failure the
        8-device CPU check surfaced for splash heads=2 and RPA kv_heads=4)."""
        w = int(getattr(case, "tp", 0) or 0)
        specs = self.tp_specs(case)
        if w < 2 or specs is None:
            return 0
        in_specs, _ = specs
        for a, spec in zip(self.abstract_inputs(case), in_specs):
            for ax, part in enumerate(spec):
                if part == "tp" and a.shape[ax] % w:
                    raise ValueError(
                        f"{self.name}/{case.name}: declared tp width {w} does not divide "
                        f"axis {ax} of size {a.shape[ax]}"
                    )
        return w

    def abstract_inputs_tp(self, case: ShapeCase, width: int):
        """Per-SHARD ShapeDtypeStructs: what the candidate is exported at for a
        TP case.

        Under shard_map the kernel is handed one device's slice, so its input
        signature is the full shape with each sharded axis divided by the mesh
        width -- which is exactly the shape a production caller's kernel sees
        inside its own shard_map. Exporting at the full shape would produce an
        artifact that shard_map can never call.
        """
        import jax

        specs = self.tp_specs(case)
        if specs is None or int(width or 0) < 2:
            return self.abstract_inputs(case)
        in_specs, _ = specs
        out = []
        for a, spec in zip(self.abstract_inputs(case), in_specs):
            shape = list(a.shape)
            for ax, part in enumerate(spec):
                if part == "tp":
                    shape[ax] = shape[ax] // width
            out.append(jax.ShapeDtypeStruct(tuple(shape), a.dtype))
        return tuple(out)

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
