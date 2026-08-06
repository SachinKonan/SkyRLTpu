"""Reward math for the arena: geomean latency ratios, the interleaved
R,C,R,C timing estimator, noise-floor gating, and speed-of-light fractions.

All functions here are pure (no jax) so the whole reward frame is testable
on synthetic latencies (DESIGN.md test layer 1).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

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


def speed_of_light_fraction(bytes_moved: int, latency_s: float,
                            chip: str) -> float | None:
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

    @property
    def score(self) -> float:
        return interleaved_score(self.pairs)

    @property
    def ref_median_s(self) -> float:
        return median([r for r, _ in self.pairs])

    @property
    def cand_median_s(self) -> float:
        return median([c for _, c in self.pairs])


def final_reward(case_timings: list[CaseTiming], noise_floor: float) -> dict:
    """Assemble the final reward: geomean over DECLARED (non-holdout) cases,
    noise-floor gated; holdout cases logged-unscored (overfit detector)."""
    declared = [t for t in case_timings if not t.holdout]
    holdout = [t for t in case_timings if t.holdout]
    if not declared:
        raise ValueError("no declared (non-holdout) case timings")
    score = geomean([t.score for t in declared])
    return {
        "score": score,
        "reward": gate_reward(score, noise_floor),
        "noise_floor": noise_floor,
        "per_case": {t.case: t.score for t in declared},
        "holdout": {t.case: t.score for t in holdout},
    }
