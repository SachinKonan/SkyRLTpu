"""Timing-protocol properties through the real grader (CPU): noise-floor
measurement, regrade stability, ref-vs-ref centering — the CPU-checkable
shadow of DESIGN.md test layer 3 (silicon invariants re-run in Phase 2)."""

from pallas_arena.judge import grader
from pallas_arena.tests import candidates as cand


def test_noise_floor_mode_ref_vs_ref(fast_grade_kwargs):
    """Ref-vs-ref through the identical interleaved protocol must center on
    1.0 (CPU jitter is looser than the 2% silicon invariant, so bounds are
    generous here; phase 2 asserts 1.00 +/- 0.02 on the chip)."""
    r = grader.measure_noise_floor("rmsnorm", smoke=True, timing_pairs=12)
    assert r["ok"], r
    assert r["noise_floor"] >= 0.0
    for case, score in r["ref_vs_ref_scores"].items():
        assert 0.5 < score < 2.0, (case, score, "ref-vs-ref wildly off 1.0")


def test_judge_provided_floor_gates_reward(fast_grade_kwargs):
    """With an enormous floor every honest grade collapses to exactly 1.0
    (tie), proving the floor gate is wired through the child."""
    kw = dict(fast_grade_kwargs, noise_floor=1000.0)
    r = grader.grade("rmsnorm", cand.HONEST_RMSNORM, **kw)
    assert r["passed"], r
    assert r["reward"] == 1.0
    assert r["noise_floor_source"] == "judge-provided"


def test_self_measured_floor_when_not_provided(fast_grade_kwargs):
    r = grader.grade("rmsnorm", cand.HONEST_RMSNORM, **fast_grade_kwargs)
    assert r["passed"], r
    assert r["noise_floor_source"] == "self-measured"
    assert r["noise_floor"] >= 0.0


def test_regrade_same_kernel_is_stable(fast_grade_kwargs):
    """Same-kernel regrade: scores agree within a very loose CPU band —
    ~40us kernels on a shared node see multi-x scheduler jitter (observed:
    0.70 vs 1.42 on a busy node). This asserts the plumbing returns
    comparable scores; the real +/-3% invariant is asserted on silicon in
    phase 4."""
    kw = dict(fast_grade_kwargs, timing_pairs=20)
    r1 = grader.grade("rmsnorm", cand.HONEST_RMSNORM, **kw)
    r2 = grader.grade("rmsnorm", cand.HONEST_RMSNORM, **kw)
    assert r1["passed"] and r2["passed"]
    ratio = r1["score"] / r2["score"]
    assert 0.25 < ratio < 4.0, (r1["score"], r2["score"])


def test_latencies_and_memory_fields_reported(fast_grade_kwargs):
    r = grader.grade("rmsnorm", cand.HONEST_RMSNORM, **fast_grade_kwargs)
    assert r["passed"]
    lat = r["latencies"]["tiny"]
    assert lat["ref_median_s"] > 0 and lat["cand_median_s"] > 0
    # peak HBM is a TPU stat; the field must exist (None on CPU is fine)
    assert "peak_hbm_bytes" in r
    # memory-bound task: speed-of-light fraction dict present (empty on CPU
    # because the chip is unknown)
    assert "speed_of_light_fracs" in r
