"""Layer-1 tests: reward math on synthetic latencies (no jax, no children)."""

import math

import pytest

from pallas_arena.judge import timing as tm


def test_geomean_basics():
    assert tm.geomean([2.0, 8.0]) == pytest.approx(4.0)
    assert tm.geomean([1.0, 1.0, 1.0]) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        tm.geomean([])
    with pytest.raises(ValueError):
        tm.geomean([1.0, -2.0])


def test_median_and_quantile():
    assert tm.median([3.0, 1.0, 2.0]) == 2.0
    assert tm.median([4.0, 1.0, 2.0, 3.0]) == 2.5
    assert tm.quantile([1.0, 2.0, 3.0, 4.0], 0.0) == 1.0
    assert tm.quantile([1.0, 2.0, 3.0, 4.0], 1.0) == 4.0
    assert tm.quantile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


def test_interleaved_estimator_cancels_linear_drift():
    """The core reason for R,C,R,C interleaving: a slow linear clock/thermal
    drift biases the naive sequential estimator but cancels pairwise."""
    n = 20
    true_ratio = 1.0
    # drift: everything gets 2x slower over the run
    drift = [1.0 + i / n for i in range(2 * n)]
    seq_ref = [100e-6 * drift[i] for i in range(n)]                # R first
    seq_cand = [100e-6 / true_ratio * drift[n + i] for i in range(n)]  # then C
    sequential_estimate = (sum(seq_ref) / n) / (sum(seq_cand) / n)

    pairs = []
    for i in range(n):
        d_r, d_c = drift[2 * i], drift[2 * i + 1]
        pairs.append((100e-6 * d_r, 100e-6 / true_ratio * d_c))
    interleaved = tm.interleaved_score(pairs)

    assert abs(interleaved - true_ratio) < 0.03
    assert abs(sequential_estimate - true_ratio) > 0.25  # badly biased
    assert abs(interleaved - true_ratio) < abs(sequential_estimate - true_ratio)


def test_pair_ratio_validation():
    with pytest.raises(ValueError):
        tm.pair_ratios([])
    with pytest.raises(ValueError):
        tm.pair_ratios([(1.0, 0.0)])


def test_noise_floor_from_ref_pairs():
    # +-5% multiplicative jitter, deterministic pattern
    pairs = []
    for i in range(40):
        j1 = 1.0 + 0.05 * math.sin(i * 1.7)
        j2 = 1.0 + 0.05 * math.cos(i * 2.3)
        pairs.append((100e-6 * j1, 100e-6 * j2))
    floor = tm.noise_floor_from_ref_pairs(pairs)
    assert 0.0 < floor < 0.25
    # a jitter-free ref-vs-ref run has (near-)zero floor
    assert tm.noise_floor_from_ref_pairs([(1e-4, 1e-4)] * 10) == 0.0


def test_gate_reward_statistical_honesty():
    # inside the floor: exactly 1.0 (ties never reward)
    assert tm.gate_reward(1.019, 0.02) == 1.0
    assert tm.gate_reward(0.995, 0.02) == 1.0
    assert tm.gate_reward(1.0, 0.0) == 1.0
    # outside the floor: passes through untouched
    assert tm.gate_reward(1.20, 0.02) == 1.20
    assert tm.gate_reward(0.70, 0.02) == 0.70
    with pytest.raises(ValueError):
        tm.gate_reward(1.0, -0.1)


def test_speed_of_light_fraction():
    # 1 GB moved in 1 ms on v6e (1.6 TB/s) = 62.5% of light
    frac = tm.speed_of_light_fraction(10**9, 1e-3, "v6e")
    assert frac == pytest.approx(1e12 / 1.6e12)
    assert tm.speed_of_light_fraction(10**9, 1e-3, "cpu") is None


def test_final_reward_holdout_logged_not_scored():
    declared1 = tm.CaseTiming("a", [(2e-4, 1e-4)] * 5)            # 2.0x
    declared2 = tm.CaseTiming("b", [(1e-4, 2e-4)] * 5)            # 0.5x
    holdout = tm.CaseTiming("h", [(5e-4, 1e-4)] * 5, holdout=True)  # 5x
    out = tm.final_reward([declared1, declared2, holdout], noise_floor=0.02)
    assert out["score"] == pytest.approx(1.0)          # geomean(2, 0.5)
    assert out["reward"] == 1.0                        # tie -> exactly 1.0
    assert set(out["per_case"]) == {"a", "b"}
    assert set(out["holdout"]) == {"h"}                # logged, unscored
    with pytest.raises(ValueError):
        tm.final_reward([holdout], noise_floor=0.0)


def test_final_reward_above_floor_passes_through():
    fast = tm.CaseTiming("a", [(3e-4, 1e-4)] * 5)
    out = tm.final_reward([fast], noise_floor=0.02)
    assert out["score"] == pytest.approx(3.0)
    assert out["reward"] == pytest.approx(3.0)


def test_case_timing_medians():
    t = tm.CaseTiming("x", [(1e-4, 2e-4), (3e-4, 4e-4), (5e-4, 6e-4)])
    assert t.ref_median_s == pytest.approx(3e-4)
    assert t.cand_median_s == pytest.approx(4e-4)
