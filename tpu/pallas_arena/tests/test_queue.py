"""Pull-queue unit tests: FIFO, leases, heartbeat, exact expiry accounting,
double-grade idempotency (phase-3 architecture)."""

import time

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from pallas_arena.judge.queue import create_queue_app  # noqa: E402


def _client(lease_timeout_s):
    return TestClient(create_queue_app(lease_timeout_s=lease_timeout_s))


def _submit(c, code="x", problem="rmsnorm"):
    r = c.post("/submit", json={"problem": problem, "code": code})
    assert r.status_code == 200
    return r.json()["work_id"]


def test_fifo_dispatch_order():
    with _client(60.0) as c:
        ids = [_submit(c, code=f"k{i}") for i in range(3)]
        got = [c.get("/work", params={"worker_id": "w"}).json()["work_id"] for _ in range(3)]
        assert got == ids
        assert c.get("/work").status_code == 204  # drained


def test_result_roundtrip_and_status():
    with _client(60.0) as c:
        wid = _submit(c)
        assert c.get(f"/result/{wid}").json() == {"done": False, "state": "queued", "attempts": 0}
        item = c.get("/work", params={"worker_id": "w1"}).json()
        r = c.post("/result", json={"lease_id": item["lease_id"], "work_id": wid, "result": {"reward": 1.5}})
        assert r.json() == {"ok": True, "duplicate": False}
        got = c.get(f"/result/{wid}").json()
        assert got["done"] and got["result"]["reward"] == 1.5
        s = c.get("/status").json()
        assert s["submitted"] == 1 and s["completed"] == 1
        assert s["queue_depth"] == 0 and s["leased"] == 0
        assert "w1" in s["workers_seen"]


def test_lease_expiry_requeues_exactly_once():
    with _client(0.6) as c:
        wid = _submit(c)
        first = c.get("/work", params={"worker_id": "w1"}).json()
        assert first["attempt"] == 1
        # not yet expired: nothing to hand out
        assert c.get("/work", params={"worker_id": "w2"}).status_code == 204
        time.sleep(0.7)
        second = c.get("/work", params={"worker_id": "w2"}).json()
        assert second["work_id"] == wid
        assert second["attempt"] == 2
        s = c.get("/status").json()
        assert s["expired_leases"] == 1 and s["requeues"] == 1


def test_heartbeat_extends_lease():
    with _client(0.8) as c:
        _submit(c)
        item = c.get("/work", params={"worker_id": "w1"}).json()
        time.sleep(0.5)
        assert c.post("/heartbeat", json={"lease_id": item["lease_id"]}).status_code == 200
        time.sleep(0.5)  # 1.0s total > 0.8 lease, but 0.5 since beat
        assert c.get("/work", params={"worker_id": "w2"}).status_code == 204
        time.sleep(0.9)  # now past the extended expiry
        assert c.get("/work", params={"worker_id": "w2"}).json()["attempt"] == 2


def test_heartbeat_after_expiry_rejected():
    with _client(0.4) as c:
        _submit(c)
        item = c.get("/work", params={"worker_id": "w1"}).json()
        time.sleep(0.5)
        assert c.post("/heartbeat", json={"lease_id": item["lease_id"]}).status_code == 404


def test_double_grade_is_idempotent():
    """Slow worker w1 loses its lease; w2 regrades and completes; w1's late
    result is accepted as a counted duplicate without clobbering."""
    with _client(0.4) as c:
        wid = _submit(c)
        lease1 = c.get("/work", params={"worker_id": "w1"}).json()["lease_id"]
        time.sleep(0.5)
        lease2 = c.get("/work", params={"worker_id": "w2"}).json()["lease_id"]
        assert c.post("/result", json={"lease_id": lease2, "work_id": wid, "result": {"reward": 2.0}}).json() == {
            "ok": True,
            "duplicate": False,
        }
        late = c.post("/result", json={"lease_id": lease1, "work_id": wid, "result": {"reward": 9.9}})
        assert late.json() == {"ok": True, "duplicate": True}
        got = c.get(f"/result/{wid}").json()
        assert got["result"]["reward"] == 2.0  # first completion wins
        assert c.get("/status").json()["duplicates"] == 1


def test_unknown_work_and_lease_404():
    with _client(60.0) as c:
        assert c.get("/result/nope").status_code == 404
        assert c.post("/result", json={"lease_id": "x", "work_id": "nope", "result": {}}).status_code == 404
        assert c.post("/heartbeat", json={"lease_id": "zz"}).status_code == 404
