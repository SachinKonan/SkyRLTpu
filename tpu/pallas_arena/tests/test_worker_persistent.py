"""Persistent-worker path (phase 3) on CPU: boot-once grading through the
sandbox-child jax.export pipeline — gates, cheaters, recalibrated goldens,
counterbalanced boot floor, warm-vs-cold cost split."""

import pytest

from pallas_arena.judge.worker import PersistentWorker
from pallas_arena.tests import candidates as cand


@pytest.fixture(scope="module")
def worker():
    w = PersistentWorker(
        "rmsnorm",
        smoke=True,
        cases=["tiny", "tiny-holdout"],
        timing_pairs=6,
        timing_warmup=1,
        determinism_runs=5,
        correctness_seeds=2,
        worker_id="pytest-worker",
    )
    report = w.boot()
    assert report["ok"], report
    return w


def test_boot_counterbalanced_floor(worker):
    r = worker.boot_report
    assert r["noise_floor"] >= 0.0
    for case, score in r["ref_vs_ref_scores"].items():
        # CPU jitter band; the ±2% assertion is phase-4 silicon
        assert 0.5 < score < 2.0, (case, score)


def test_honest_kernel_passes_with_warm_cost_split(worker):
    r = worker.grade_code(cand.HONEST_RMSNORM)
    assert r["passed"], r
    assert r["gate"] == "all"
    assert 0.2 < r["score"] < 5.0
    assert set(r["per_case"]) == {"tiny"}
    assert set(r["holdout"]) == {"tiny-holdout"}
    # the cost split exists: export (sandbox child) / artifact load /
    # candidate compile-warm / steady-state chip time — and the warm chip
    # time excludes every compile (the ~4s phase-4 target metric)
    assert r["export_s"] > 0 and r["warm_chip_s"] > 0
    assert r["candidate_compile_s"] > 0
    assert r["warm_chip_s"] < r["export_s"] + r["load_s"] + r["candidate_compile_s"]


def test_recalibrated_goldens_pass(worker):
    """The phase-2 tolerance casualties, re-armed: an overflow-robust
    unjitted kernel must now pass the full gate battery (calibration spans
    honest implementation variants, not the reference alone)."""
    r = worker.grade_code(cand.UNJITTED_ROBUST_RMSNORM)
    assert r["passed"], r


def test_cached_output_constant_baked_into_graph_fails(worker):
    """In the export path a cached-output cheat becomes a baked constant —
    fresh worker-side seeds still kill it.

    Gate depends on the shape it cached. This suite's smoke case is `tiny`
    = (32, 64), which is exactly the constant the cheater hard-codes, so
    here it survives export and dies at `correctness`. On silicon (phase 4)
    the declared signatures are 8192x4096 and larger, the constant cannot
    broadcast, and jax.export refuses to serialize it at all — rejected one
    gate EARLIER. Both outcomes are rejections; neither is weaker."""
    r = worker.grade_code(cand.CACHED_OUTPUT_RMSNORM)
    assert not r["passed"]
    assert r["gate"] in ("correctness", "aot_export"), r


def test_constant_output_shape_correct_cheat_dies_at_correctness(worker):
    """The shape-coincidence above meant the "ignores its inputs, returns a
    plausible answer" cheat was never actually exercised against the gate
    that is supposed to catch it. This variant broadcasts to whatever shape
    it is handed, so it exports cleanly at EVERY declared signature and has
    to be killed by fresh hidden seeds."""
    r = worker.grade_code(cand.CONSTANT_OUTPUT_RMSNORM)
    assert not r["passed"]
    assert r["gate"] == "correctness", r


def test_grade_budget_rejects_a_wedging_candidate(worker):
    """Phase-4 lesson: an unbounded grade turns one pathological kernel into
    a fleet-wide poison pill. A 24-deep unjitted waste chain held the judge
    for 19+ minutes and never returned, so its lease expired and the item
    requeued — onto the next judge, which would wedge identically. The
    budget bounds the blast radius to one candidate. It is checked between
    phases, so an honest kernel graded under an impossible budget must come
    back rejected at gate `budget` rather than hang."""
    prev = worker.grade_budget_s
    worker.grade_budget_s = 1e-9
    try:
        r = worker.grade_code(cand.HONEST_RMSNORM + "\n# budget probe\n")
    finally:
        worker.grade_budget_s = prev
    assert not r["passed"]
    assert r["gate"] == "budget", r
    assert "budget" in r["violations"][0]


def test_obfuscated_import_dies_in_export_child(worker):
    r = worker.grade_code(cand.OBFUSCATED_IMPORT_RMSNORM)
    assert not r["passed"]
    assert r["gate"] in ("poison_stub", "aot_export")


def test_wrong_grad_caught_through_exported_grad(worker):
    r = worker.grade_code(cand.WRONG_GRAD_RMSNORM)
    assert not r["passed"]
    assert r["gate"] == "gradient"


def test_wrong_eps_still_caught_under_wider_calibration(worker):
    """Recalibration widens the margin to honest-variant spread; the
    wrong-epsilon kernel must STILL fail on the small-magnitude vector."""
    r = worker.grade_code(cand.WRONG_EPS_RMSNORM)
    assert not r["passed"]
    assert r["gate"] == "correctness"


def test_timer_tamperer_structurally_neutralized(worker):
    """No candidate python ever runs in the worker: the tamperer's patches
    die with the export child, its compiled graph is honest-but-slow."""
    r = worker.grade_code(cand.TIMER_TAMPERER_RMSNORM)
    assert r["passed"], r
    assert r["score"] < 0.9, r["score"]


def test_split_personality_structurally_neutralized(worker):
    """Python call-counting is baked away at trace time: the exported graph
    is permanently the honest branch, so the fast-garbage phase can never
    exist on the judge. The kernel simply grades as its honest self."""
    r = worker.grade_code(cand.SPLIT_PERSONALITY_RMSNORM)
    assert r["passed"], r
    assert r["gate"] == "all"


def test_worker_cache_hit(worker, tmp_path):
    from pallas_arena.judge.cache import RewardCache

    worker.cache = RewardCache(str(tmp_path / "wc"))
    try:
        r1 = worker.grade_code(cand.HONEST_RMSNORM)
        assert not r1.get("cache_hit")
        r2 = worker.grade_code(cand.HONEST_RMSNORM)
        assert r2["cache_hit"]
        assert r2["reward"] == r1["reward"]
    finally:
        worker.cache = None
