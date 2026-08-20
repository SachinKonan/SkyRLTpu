"""Reward math for the arena: geomean latency ratios, the interleaved
R,C,R,C timing estimator, noise-floor gating, and speed-of-light fractions.

All functions here are pure (no jax) so the whole reward frame is testable
on synthetic latencies (DESIGN.md test layer 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# Peak HBM bandwidth per chip generation, bytes/s (QuACK-style yardstick for
# memory-bound tasks; DESIGN.md "Reward frame").
SPEED_OF_LIGHT_BYTES_PER_S = {
    "v6e": 1.6e12,
    "v5p": 2.8e12,
}


def geomean(xs: list[float]) -> float:
    if not xs:
        raise ValueError("geomean of empty list")
    if any(x <= 0 for x in xs):
        raise ValueError(f"geomean requires positive values, got {xs}")
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


def pair_ratios(pairs: list[tuple[float, float]]) -> list[float]:
    """Per-pair latency ratios reference/candidate from interleaved timing.

    Each pair (r_lat, c_lat) is one adjacent R,C invocation pair, so slow
    linear clock/thermal drift hits both legs of a pair nearly equally and
    cancels in the ratio — the reason the protocol interleaves rather than
    timing 20 R then 20 C.
    """
    if not pairs:
        raise ValueError("no timing pairs")
    for r, c in pairs:
        if r <= 0 or c <= 0:
            raise ValueError(f"non-positive latency in pair ({r}, {c})")
    return [r / c for r, c in pairs]


def median(xs: list[float]) -> float:
    s = sorted(xs)
    n = len(s)
    if n == 0:
        raise ValueError("median of empty list")
    mid = n // 2
    return s[mid] if n % 2 else 0.5 * (s[mid - 1] + s[mid])


def quantile(xs: list[float], q: float) -> float:
    """Simple linear-interpolation quantile (numpy 'linear' method)."""
    s = sorted(xs)
    if not s:
        raise ValueError("quantile of empty list")
    if len(s) == 1:
        return s[0]
    pos = q * (len(s) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def interleaved_score(pairs: list[tuple[float, float]]) -> float:
    """Median of per-pair ref/candidate ratios (median-of-N estimator)."""
    return median(pair_ratios(pairs))


def amortized_call(fn, args, repeats: int):
    """Call ``fn(*args)`` ``repeats`` times inside ONE jit, chained through
    ``jax.lax.optimization_barrier`` so XLA cannot collapse the repeats.

    WHY. Our per-pair timing is wallclock (`perf_counter` + `block_until_ready`),
    which includes Python dispatch. With overhead `c` and true times `a`/`b` the
    measured ratio is (a+c)/(b+c), NOT a/b -- so every reward is compressed
    TOWARD 1.0, understating real speedups and overstating real slowdowns. The
    counterbalanced alternation cancels drift and first-position penalty; it
    cannot cancel a constant added to both legs.

    tokamax's harness sidesteps this by defaulting to device-level profiling on
    TPU (`hermetic_xprof`; wallclock is only its fallback). Rather than take a
    profiling dependency into the judge, amortize: k chained calls per dispatch
    divides the overhead by k. The barrier is load-bearing -- without it XLA
    CSEs the identical calls into one and the timing silently measures 1/k of
    the work.
    """
    import jax

    def body(*a):
        out = fn(*a)
        for _ in range(repeats - 1):
            a = jax.lax.optimization_barrier(a)
            out = fn(*a)
        return out

    return body


def device_timer(fn, args):
    """Device-time one call of ``fn(*args)`` in milliseconds, or None.

    Copies tokamax's TPU default (`hermetic_xprof`): profile the call and take
    the DISJOINT INTERVAL UNION of XLA op intervals -- total active device
    time, overlapping ops merged -- instead of a Python stopwatch. On TPU the
    instrumentation is added at compile time with near-zero overhead, so this
    measures the kernel rather than the kernel plus dispatch.

    Why we want it: our reward is a RATIO. A Python stopwatch adds the same
    dispatch cost `c` to both legs, so a true a/b is measured as (a+c)/(b+c)
    and every score is compressed TOWARD 1.0 -- real speedups understated,
    real slowdowns overstated, worst at our fastest shapes (RPA's baseline is
    0.21 ms, where c is not small).

    Returns None when profiling is unavailable (CPU, missing xprof, any
    failure) so the caller falls back to wallclock rather than failing a grade.
    """
    try:
        import jax

        if jax.default_backend() == "cpu":
            return None
        from tokamax._src.benchmarking import XprofProfileSession
    except Exception:  # noqa: BLE001 -- no tokamax/xprof on this host
        return None
    try:
        import datetime

        import jax

        jax.block_until_ready(fn(*args))  # warm; never timed
        with XprofProfileSession(hermetic=True, use_jax_profiler=True) as prof:
            jax.block_until_ready(fn(*args))
        return prof.total_op_time / datetime.timedelta(milliseconds=1)
    except Exception:  # noqa: BLE001 -- profiling must never fail a grade
        return None


def device_timing_available() -> bool:
    """Probe once at boot: can this judge device-time at all?"""
    try:
        import jax
        import jax.numpy as jnp

        if jax.default_backend() == "cpu":
            return False
        x = jnp.ones((256, 256), jnp.float32)
        return device_timer(lambda a: a @ a, (x,)) is not None
    except Exception:  # noqa: BLE001
        return False


def counterbalanced_pair_device(i, ref_fn, cand_fn, args, dev_timer):
    """Device-timed counterbalanced pair: same alternation as the wallclock
    version (so the first-position penalty still cancels), but each leg's
    latency comes from the profiler's active-device time rather than a Python
    stopwatch. Returns None if either leg fails to profile, so the caller can
    fall back for that pair instead of dropping the case."""
    if i % 2 == 0:
        r = dev_timer(ref_fn, args)
        c = dev_timer(cand_fn, args)
    else:
        c = dev_timer(cand_fn, args)
        r = dev_timer(ref_fn, args)
    if not r or not c or r <= 0 or c <= 0:
        return None
    return (r, c)


def tp_width(shard_dim: int, n_devices: int) -> int:
    """Largest power-of-two mesh width that DIVIDES the sharded axis.

    The TP axis must divide by the device count or device_put raises
    IndivisibleError -- measured on the 8-simulated-device CPU check, where
    splash's heads=2 and RPA's kv_heads=4 cases both failed against a width-8
    mesh. Sizing the mesh to the case (rather than assuming all 8 chips) keeps
    every shape gradeable and reports the width actually used, which is the
    honest thing to report anyway: "4-way sharded" is a different claim from
    "8-way sharded"."""
    w = min(n_devices, shard_dim)
    while w > 1 and (shard_dim % w or n_devices % w):
        w -= 1
    return max(1, w)


def make_mesh(n_devices: int | None = None):
    """Single-axis 'tp' mesh over the judge's local devices, or None if there
    is only one. v6e-8 is a SINGLE host (8 chips, 1 TensorCore each), so this
    never touches the multi-host rendezvous that requires every host to join."""
    import jax
    from jax.sharding import Mesh

    devs = jax.local_devices()
    if n_devices:
        devs = devs[: int(n_devices)]
    if len(devs) < 2:
        return None
    return Mesh(devs, ("tp",))


def shard_mapped(fn, mesh, in_specs, out_spec):
    """Wrap ``fn`` so each device runs it on its own shard.

    shard_map rather than plain jit+shardings on purpose: the candidate wrote a
    Pallas kernel, and under shard_map it receives a per-device shard and runs
    unchanged -- which is how splash (`head_shards`) and megablox
    (`group_offset`) are used in production. GSPMD auto-partitioning of a
    hand-written pallas_call is not what any real caller does, so grading it
    that way would measure the wrong thing.
    """
    from jax.experimental.shard_map import shard_map

    return shard_map(fn, mesh=mesh, in_specs=in_specs, out_specs=out_spec, check_rep=False)


def shard_inputs(inputs, mesh, in_specs):
    """Place inputs on the mesh with the task's declared specs."""
    import jax
    from jax.sharding import NamedSharding

    return tuple(
        jax.device_put(x, NamedSharding(mesh, spec)) for x, spec in zip(inputs, in_specs)
    )


def counterbalanced_pair(i, run_ref, run_cand, perf, block):
    """Run ONE timed pair with alternating position order: R,C on even i,
    C,R on odd i.

    Phase-2 shakedown finding: with a fixed R-first order, ref-vs-ref
    graded 1.019-1.053 (FAILED the 1.00±2% invariant) — the first-position
    leg after fresh on-device input generation is systematically slower.
    Alternating which role runs first spreads that fixed penalty equally
    over both roles, so it cancels in the median of per-pair ratios while
    keeping the interleaving's drift cancellation.

    Returns ((ref_latency_s, cand_latency_s), ref_out, cand_out).
    """
    if i % 2 == 0:
        t0 = perf()
        a = run_ref()
        block(a)
        t1 = perf()
        b = run_cand()
        block(b)
        t2 = perf()
        return (t1 - t0, t2 - t1), a, b
    t0 = perf()
    b = run_cand()
    block(b)
    t1 = perf()
    a = run_ref()
    block(a)
    t2 = perf()
    return (t2 - t1, t1 - t0), a, b


def noise_floor_from_ref_pairs(pairs: list[tuple[float, float]]) -> float:
    """Noise floor from a ref-vs-ref run of the same interleaved protocol.

    Floor = p95 of |ratio - 1| over ref/ref pairs: the score deviation the
    protocol produces when there is NO true speedup. Logged per chip at
    suite/boot time (DESIGN.md test layer 3 + statistical honesty).
    """
    ratios = pair_ratios(pairs)
    return quantile([abs(r - 1.0) for r in ratios], 0.95)


def gate_reward(score: float, noise_floor: float) -> float:
    """Apply statistical honesty: reward > 1.0 only when the measured
    speedup exceeds the per-chip noise floor; ties score exactly 1.0.

    Scores inside [1 - floor, 1 + floor] collapse to exactly 1.0; scores
    outside pass through unchanged (a genuine slowdown stays a slowdown).
    """
    if noise_floor < 0:
        raise ValueError("noise floor must be >= 0")
    if abs(score - 1.0) <= noise_floor:
        return 1.0
    return score


def speed_of_light_fraction(bytes_moved: int, latency_s: float, chip: str) -> float | None:
    """Fraction of peak HBM bandwidth achieved (memory-bound tasks only).

    Logged alongside the latency-ratio score to expose weak-baseline
    pseudo-wins; returns None off-TPU / unknown chips.
    """
    bw = SPEED_OF_LIGHT_BYTES_PER_S.get(chip)
    if bw is None or latency_s <= 0:
        return None
    return (bytes_moved / latency_s) / bw


@dataclass
class CaseTiming:
    """Timing result for one shape case."""

    case: str
    pairs: list[tuple[float, float]] = field(default_factory=list)
    holdout: bool = False
    # BLIND cases are measured and reported but NEVER scored, in either reward
    # mode. See final_reward.
    blind: bool = False

    @property
    def score(self) -> float:
        return interleaved_score(self.pairs)

    @property
    def ref_median_s(self) -> float:
        return median([r for r, _ in self.pairs])

    @property
    def cand_median_s(self) -> float:
        return median([c for _, c in self.pairs])


def final_reward(case_timings: list[CaseTiming], noise_floor: float, *, general: bool = False) -> dict:
    """Assemble the final reward, noise-floor gated.

    Two modes, because they want opposite things from the holdout:

    OURS_SPECIFIC (``general=False``, the historical behaviour): geomean over
    DECLARED cases only; holdout logged-unscored. Correct when the goal is a
    kernel for OUR production shape -- hyper-specializing to it is the point,
    and penalizing the holdout would penalize exactly that.

    GENERAL_OPTIMIZATION (``general=True``): geomean over EVERY case, holdout
    included, because generality IS the objective. Without this a kernel can
    hardcode the two shapes it was shown, be broken at the deliberately
    non-divisible holdout, and still collect full reward -- the evidence
    sitting unused in the ``holdout`` field. The result is labelled
    ``reward_kind`` so a general number is never mistaken for a historical one.
    """
    # BLIND cases are removed from BOTH candidate pools before anything is
    # scored. They are the validation set, and a validation set the model is
    # paid for stops measuring anything: general mode deliberately scores the
    # holdout (so a kernel cannot hardcode the shapes it was shown), which is
    # correct but leaves nothing unoptimized to check generalization against.
    # `blind` is that third category -- measured, reported, never rewarded.
    blind = [t for t in case_timings if t.blind]
    pool = [t for t in case_timings if not t.blind]
    declared = [t for t in pool if not t.holdout]
    holdout = [t for t in pool if t.holdout]
    if not declared:
        raise ValueError("no declared (non-holdout, non-blind) case timings")
    scored = pool if general else declared
    score = geomean([t.score for t in scored])
    return {
        "score": score,
        "reward": gate_reward(score, noise_floor),
        "reward_kind": "general" if general else "ours",
        "n_scored_cases": len(scored),
        "noise_floor": noise_floor,
        "per_case": {t.case: t.score for t in declared},
        "holdout": {t.case: t.score for t in holdout},
        "holdout_scored": bool(general and holdout),
        # The validation signal: what the BLIND cases scored, reported so a
        # tier-1-vs-blind gap is visible without ever entering the reward.
        "n_blind_cases": len(blind),
        "blind_score": geomean([t.score for t in blind]) if blind else None,
        "blind_per_case": {t.case: t.score for t in blind},
    }


# The BACKWARD-FOLDED total: the training scalar for RL on has_bwd tasks.
#
# The backward joins the geomean as ONE MORE CASE (weight 1/(n+1)) -- one
# number, one domain, TriMul's one-scalar discipline -- with an ABSENCE
# FLOOR instead of 0: a literal zero in a geomean zeroes the total, which
# silently re-creates the hard gradient gate that flattened 8/8 splash
# winners (flat reward = no RL signal). max(noise_floor, 0.05) preserves
# the ordering that matters -- no backward (~25-35% haircut at our case
# counts) < slow-but-correct backward < fast backward -- without a cliff.
GRAD_ABSENT_FLOOR = 0.05


def fold_grad_reward(reward_frame: dict, grad_score: float | None,
                     noise_floor: float, n_scored: int) -> float:
    """Fold the backward into the forward geomean as one more case."""
    fwd = reward_frame["score"]
    comp = grad_score if grad_score else max(noise_floor, GRAD_ABSENT_FLOOR)
    total = (fwd ** n_scored * comp) ** (1.0 / (n_scored + 1))
    return gate_reward(total, noise_floor)
