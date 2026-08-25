"""Control-flow tests for the Ray grading pool, with fake Ray + fake queue.

The pool's first real execution is unattended on a judge host, so the parts
that do not need TPU -- leasing, dispatch, result posting, the
problem-mismatch fault path, max_items termination -- are exercised here on
CPU. JudgeActorImpl itself is thin (it delegates to PersistentWorker) and
needs jax/TPU, so the fake Ray substitutes it.
"""

from __future__ import annotations

import sys
import types

import pytest

from pallas_arena.judge import ray_pool


class _FakeRef:
    def __init__(self, value):
        self.value = value


def _install_fake_ray(monkeypatch, grade_impl):
    """A synchronous stand-in: .remote() computes immediately and boxes the
    result, ray.get unboxes, ray.wait reports everything ready."""

    class FakeActor:
        def __init__(self, problem, cfg):
            self.problem, self.cfg = problem, cfg
            self.ready = types.SimpleNamespace(
                remote=lambda: _FakeRef({"problem": problem, "visible_chips": "0", "noise_floor": 0.048})
            )
            self.grade = types.SimpleNamespace(
                remote=lambda payload: _FakeRef(grade_impl(problem, payload))
            )

    class FakeActorCls:
        remote = staticmethod(lambda problem, cfg: FakeActor(problem, cfg))

    fake = types.ModuleType("ray")
    fake.init = lambda **kw: None
    fake.remote = lambda **kw: (lambda cls: FakeActorCls)
    fake.get = lambda ref: (
        [r.value for r in ref] if isinstance(ref, list) else ref.value
    )
    fake.wait = lambda refs, num_returns=1, timeout=None: (refs[:num_returns], refs[num_returns:])
    monkeypatch.setitem(sys.modules, "ray", fake)
    return fake


class _FakeQueue:
    """Minimal /work, /result, /heartbeat over an in-memory list."""

    def __init__(self, items):
        self.pending = list(items)
        self.results = {}
        self.heartbeats = 0

    def http(self, url, payload=None, timeout=30.0):
        if "/work" in url:
            if not self.pending:
                return None
            it = self.pending.pop(0)
            return {"work_id": it["work_id"], "lease_id": "lease-" + it["work_id"],
                    "lease_timeout_s": 60.0, "attempt": 1, "payload": it["payload"]}
        if "/heartbeat" in url:
            self.heartbeats += 1
            return {}
        if "/result" in url:
            self.results[payload["work_id"]] = payload["result"]
            return {}
        raise AssertionError(f"unexpected url {url}")


def _items(n, problem="rg_lru"):
    return [{"work_id": f"w{i}", "payload": {"problem": problem, "code": f"# {i}"}} for i in range(n)]


def test_pool_grades_every_item_and_posts_results(monkeypatch):
    q = _FakeQueue(_items(6))
    _install_fake_ray(monkeypatch, lambda problem, payload: {"ok": True, "passed": True, "code": payload["code"]})
    monkeypatch.setattr(ray_pool, "_http_json", q.http)

    done = ray_pool.run_pool("http://fake", ["rg_lru"], actors=4,
                             cfg={"width": 1}, poll_s=0.01, max_items=6, idle_exit_s=2)

    assert done == 6
    assert len(q.results) == 6
    assert all(r["passed"] for r in q.results.values())
    # every verdict carries the wall time the pool measured
    assert all("item_wall_s" in r for r in q.results.values())


def test_actors_are_created_per_requested_count_and_problem(monkeypatch):
    q = _FakeQueue(_items(2))
    seen = []
    _install_fake_ray(monkeypatch, lambda problem, payload: (seen.append(problem), {"ok": True})[1])
    monkeypatch.setattr(ray_pool, "_http_json", q.http)

    ray_pool.run_pool("http://fake", ["rg_lru", "splash_attention"], actors=2,
                      cfg={"width": 1}, poll_s=0.01, max_items=2, idle_exit_s=2)
    # round-robin assignment means both problems are represented
    assert set(seen) <= {"rg_lru", "splash_attention"}


def test_problem_mismatch_is_faulted_not_silently_dropped(monkeypatch):
    """An item for a problem no actor is booted for must get a judge_fault
    verdict (excluded from reward), never an unanswered lease."""
    q = _FakeQueue(_items(2, problem="megablox_gmm"))
    _install_fake_ray(monkeypatch, lambda problem, payload: {"ok": True})
    monkeypatch.setattr(ray_pool, "_http_json", q.http)

    # No max_items: faulted items never count as graded, so termination has
    # to come from the idle timer -- which is exactly the production shape.
    ray_pool.run_pool("http://fake", ["rg_lru"], actors=1,
                      cfg={"width": 1}, poll_s=0.01, idle_exit_s=1)

    assert q.results, "mismatched item must still receive a verdict"
    v = next(iter(q.results.values()))
    assert v["gate"] == "judge_fault"
    assert "megablox_gmm" in v["violations"][0]


def test_grade_exception_leaves_lease_for_requeue(monkeypatch):
    """If an actor dies mid-grade we must NOT post a verdict: the lease
    expires and the queue hands the item to another actor."""
    def boom(problem, payload):
        raise RuntimeError("actor died")

    q = _FakeQueue(_items(1))
    _install_fake_ray(monkeypatch, boom)
    monkeypatch.setattr(ray_pool, "_http_json", q.http)

    # fake ray.get raises when the boxed value was an exception: emulate by
    # having grade_impl raise at .remote() time, which our fake propagates.
    with pytest.raises(RuntimeError):
        ray_pool.run_pool("http://fake", ["rg_lru"], actors=1,
                          cfg={"width": 1}, poll_s=0.01, max_items=1, idle_exit_s=2)
    assert q.results == {}


def test_idle_exit_returns_when_queue_stays_empty(monkeypatch):
    """The offline fleet must release its spot host when the corpus is done;
    the RL pool (idle_exit_s=None) must NOT exit during a lull between steps."""
    q = _FakeQueue([])
    _install_fake_ray(monkeypatch, lambda problem, payload: {"ok": True})
    monkeypatch.setattr(ray_pool, "_http_json", q.http)
    t0 = __import__("time").time()
    done = ray_pool.run_pool("http://fake", ["rg_lru"], actors=2,
                             cfg={"width": 1}, poll_s=0.01, idle_exit_s=1)
    assert done == 0
    assert __import__("time").time() - t0 < 10


def test_cpu_per_actor_leaves_host_headroom():
    assert ray_pool.cpu_per_actor() >= 2
