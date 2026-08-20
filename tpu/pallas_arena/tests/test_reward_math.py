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
    seq_ref = [100e-6 * drift[i] for i in range(n)]  # R first
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


def test_counterbalanced_pair_cancels_first_position_penalty():
    """Phase-2 finding: the first leg of each timed pair pays a fixed
    penalty (ref-vs-ref graded 1.02-1.05). With a synthetic 25% first-slot
    penalty, the fixed R,C order inherits the full bias while the
    counterbalanced R,C/C,R order collapses it to the tiny arithmetic-vs-
    geometric residual (~bias^2/2)."""
    clock = [0.0]

    def perf():
        return clock[0]

    def block(x):
        return x

    state = {"first_in_pair": True}

    def run():
        cost = 1e-3 * (1.25 if state["first_in_pair"] else 1.0)
        state["first_in_pair"] = False
        clock[0] += cost
        return None

    fixed_pairs = []
    for _ in range(20):
        state["first_in_pair"] = True
        t0 = perf()
        run()
        t1 = perf()
        run()
        t2 = perf()
        fixed_pairs.append((t1 - t0, t2 - t1))

    cb_pairs = []
    for i in range(20):
        state["first_in_pair"] = True
        pair, _, _ = tm.counterbalanced_pair(i, run, run, perf, block)
        cb_pairs.append(pair)

    fixed = tm.interleaved_score(fixed_pairs)
    cb = tm.interleaved_score(cb_pairs)
    assert abs(fixed - 1.25) < 0.01  # full first-position bias
    assert abs(cb - 1.0) < 0.03  # collapsed to the residual
    assert abs(cb - 1.0) < abs(fixed - 1.0) / 5


def test_counterbalanced_pair_role_mapping():
    """Latencies map to the correct roles in both orders."""
    clock = [0.0]

    def perf():
        return clock[0]

    def block(x):
        return x

    def make(cost, tagname):
        def _r():
            clock[0] += cost
            return tagname

        return _r

    ref, cand = make(2e-3, "ref"), make(1e-3, "cand")
    for i in (0, 1):  # both orders
        (r_lat, c_lat), r_out, c_out = tm.counterbalanced_pair(i, ref, cand, perf, block)
        assert (r_out, c_out) == ("ref", "cand")
        assert r_lat == pytest.approx(2e-3)
        assert c_lat == pytest.approx(1e-3)


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
    declared1 = tm.CaseTiming("a", [(2e-4, 1e-4)] * 5)  # 2.0x
    declared2 = tm.CaseTiming("b", [(1e-4, 2e-4)] * 5)  # 0.5x
    holdout = tm.CaseTiming("h", [(5e-4, 1e-4)] * 5, holdout=True)  # 5x
    out = tm.final_reward([declared1, declared2, holdout], noise_floor=0.02)
    assert out["score"] == pytest.approx(1.0)  # geomean(2, 0.5)
    assert out["reward"] == 1.0  # tie -> exactly 1.0
    assert set(out["per_case"]) == {"a", "b"}
    assert set(out["holdout"]) == {"h"}  # logged, unscored
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


# ------------------------------------------------------- blind validation set
def test_blind_cases_are_never_scored_in_either_mode():
    """A blind case is measured and reported but pays nothing, in BOTH reward
    modes.

    This is the whole reason `blind` exists. `general_mode` scores the holdout
    on purpose -- otherwise a kernel hardcodes the shapes it was shown -- but
    that leaves no case the model is not optimizing, and therefore no way to
    tell a kernel that learned attention from one that learned our case list.
    A blind case can only answer that question while it stays unrewarded.
    """
    from pallas_arena.judge.timing import CaseTiming, final_reward

    fast = [(2.0, 1.0)] * 8   # candidate 2x faster than ref -> score ~2.0
    slow = [(1.0, 4.0)] * 8   # candidate 4x slower          -> score ~0.25
    timings = [
        CaseTiming(case="declared", pairs=fast),
        CaseTiming(case="holdout", pairs=fast, holdout=True),
        CaseTiming(case="blind", pairs=slow, blind=True),
    ]
    for general in (False, True):
        out = final_reward(timings, noise_floor=0.0, general=general)
        # the catastrophic blind case must not drag the reward down at all
        assert out["score"] > 1.5, (general, out)
        assert out["n_blind_cases"] == 1
        assert out["blind_score"] < 0.5, out
        assert "blind" in out["blind_per_case"]
        # and it is not counted among the scored cases
        assert out["n_scored_cases"] == (2 if general else 1), out


def test_blind_only_case_set_is_rejected():
    """Scoring needs at least one declared case; an all-blind set is a
    misconfiguration and must fail loudly rather than score vacuously."""
    import pytest

    from pallas_arena.judge.timing import CaseTiming, final_reward

    with pytest.raises(ValueError):
        final_reward([CaseTiming(case="b", pairs=[(1.0, 1.0)], blind=True)], noise_floor=0.0)


# ------------------------------------------------------ backward geomean fold
def test_fold_grad_reward_orders_none_slow_fast():
    """The fold's whole job: no-backward < slow-correct-backward <
    fast-backward, with no zero cliff (a literal 0 in the geomean would
    re-create the hard gradient gate that produced a flat, unlearnable
    reward surface)."""
    from pallas_arena.judge.timing import fold_grad_reward

    frame = {"score": 0.5}
    none_ = fold_grad_reward(frame, None, noise_floor=0.02, n_scored=6)
    slow = fold_grad_reward(frame, 0.2, noise_floor=0.02, n_scored=6)
    fast = fold_grad_reward(frame, 0.9, noise_floor=0.02, n_scored=6)
    assert 0 < none_ < slow < fast, (none_, slow, fast)
    # absence is a haircut, not an execution
    assert none_ > 0.2 * frame["score"], none_
    # and a strong backward on a strong forward can still clear parity gating
    assert fold_grad_reward({"score": 1.2}, 1.2, 0.02, 6) > 1.0


def test_fold_grad_reward_weight_is_one_case():
    """The backward carries exactly 1/(n+1) weight -- folding it as a case,
    not as a separate weighted term."""
    import math

    from pallas_arena.judge.timing import fold_grad_reward

    n = 7
    got = fold_grad_reward({"score": 0.4}, 0.8, noise_floor=0.0, n_scored=n)
    want = math.exp((n * math.log(0.4) + math.log(0.8)) / (n + 1))
    assert abs(got - want) < 1e-12
