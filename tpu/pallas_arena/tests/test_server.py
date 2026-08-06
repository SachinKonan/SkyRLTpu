"""FastAPI FIFO judge server tests: launch-flag problem lock (anti-cheat),
FIFO ordering, health, end-to-end grading through the queue."""

import threading

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pallas_arena.judge.server import create_app  # noqa: E402
from pallas_arena.tests import candidates as cand  # noqa: E402


def _app(**kw):
    defaults = dict(
        workers=1,
        smoke=True,
        measure_floor_on_boot=False,
        grade_overrides=dict(
            cases=["tiny"],
            timing_pairs=4,
            timing_warmup=1,
            correctness_seeds=1,
            determinism_runs=3,
            timeout_s=120.0,
        ),
    )
    defaults.update(kw)
    return create_app("rmsnorm", **defaults)


def test_rejects_other_problem_types_400():
    with TestClient(_app()) as client:
        resp = client.post("/grade", json={"problem": "splash_attention", "code": "def kernel(): pass"})
        assert resp.status_code == 400
        assert "rmsnorm" in resp.json()["detail"]


def test_rejects_bad_mode_400():
    with TestClient(_app()) as client:
        resp = client.post("/grade", json={"problem": "rmsnorm", "code": "x", "mode": "noise_floor"})
        assert resp.status_code == 400


def test_healthz_reports_state():
    with TestClient(_app()) as client:
        h = client.get("/healthz").json()
        assert h["ok"] and h["problem"] == "rmsnorm"
        assert h["workers"] == 1
        assert h["queue_depth"] == 0


def test_grade_end_to_end_honest_and_cheater():
    with TestClient(_app()) as client:
        r = client.post("/grade", json={"problem": "rmsnorm", "code": cand.HONEST_RMSNORM}).json()
        assert r["passed"], r
        assert r["worker"] == 0
        r2 = client.post("/grade", json={"problem": "rmsnorm", "code": cand.CACHED_OUTPUT_RMSNORM}).json()
        assert not r2["passed"]
        assert r2["gate"] == "correctness"
        h = client.get("/healthz").json()
        assert h["graded_total"] == 2


def test_fifo_single_worker_completion_order():
    """Request ids are assigned at enqueue; a single worker must complete
    them in exactly that order."""
    app = _app(
        grade_overrides=dict(
            cases=["tiny"],
            mode="gates",
            timing_pairs=2,
            timing_warmup=0,
            correctness_seeds=1,
            determinism_runs=2,
            timeout_s=120.0,
        )
    )
    results = []
    with TestClient(app) as client:

        def post():
            r = client.post("/grade", json={"problem": "rmsnorm", "code": cand.HONEST_RMSNORM, "mode": "gates"})
            results.append(r.json())

        threads = [threading.Thread(target=post) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        order = app.state.arena["completed_order"]
    assert len(order) == 3
    assert order == sorted(order), f"FIFO violated: {order}"
    assert all(r["passed"] for r in results)


def test_boot_noise_floor_measured_and_used():
    app = _app(measure_floor_on_boot=True)
    with TestClient(app) as client:
        h = client.get("/healthz").json()
        assert h["noise_floors"][0] is not None
        assert h["noise_floors"][0] >= 0.0
        assert h["boot_ref_vs_ref"][0] is not None
        r = client.post("/grade", json={"problem": "rmsnorm", "code": cand.HONEST_RMSNORM}).json()
        assert r["passed"]
        assert r["noise_floor_source"] == "judge-provided"
