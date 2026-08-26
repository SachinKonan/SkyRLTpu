"""Control-flow tests for the Ray grading pool (task mode), with fake Ray +
fake queue.

The pool's first real execution is unattended on a judge host, so the parts
that do not need TPU -- leasing, width-aware dispatch, the chip budget,
result posting, the problem-mismatch fault path, task-death requeue,
idle-exit -- are exercised here on CPU. grade_one itself needs jax/TPU, so
the fake Ray substitutes it.
"""

from __future__ import annotations

import sys
import types

import pytest

from pallas_arena.judge import ray_pool


class _FakeRef:
    def __init__(self, fn):
        self._fn = fn


def _install_fake_ray(monkeypatch, grade_impl):
    """Synchronous stand-in: .remote() captures args, ray.get executes."""

    class FakeRemoteFn:
        def __init__(self, resources=None):
            self.resources = resources or {}

        def options(self, resources=None, **kw):
            return FakeRemoteFn(resources)

        def remote(self, problem, payload, cfg):
            res = self.resources
            return _FakeRef(lambda: grade_impl(problem, payload, res))

    fake = types.ModuleType("ray")
    fake.init = lambda **kw: None
    fake.remote = lambda **kw: (lambda fn: FakeRemoteFn())
    fake.get = lambda ref: ref._fn()
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


def _items(n, problem="rg_lru", **extra):
    return [{"work_id": f"w{i}", "payload": {"problem": problem, "code": f"# {i}", **extra}}
            for i in range(n)]


def _run(monkeypatch, items, grade_impl, **kw):
    q = _FakeQueue(items)
    _install_fake_ray(monkeypatch, grade_impl)
    monkeypatch.setattr(ray_pool, "_http_json", q.http)
    kw.setdefault("cfg", {"width": 1})
    kw.setdefault("poll_s", 0.01)
    done = ray_pool.run_pool("http://fake", kw.pop("problems", ["rg_lru"]),
                             chips=kw.pop("chips", 4), **kw)
    return q, done


def test_pool_grades_every_item_and_posts_results(monkeypatch):
    q, done = _run(monkeypatch, _items(6),
                   lambda p, payload, res: {"ok": True, "passed": True},
                   max_items=6, idle_exit_s=2)
    assert done == 6
    assert len(q.results) == 6
    assert all(r["passed"] for r in q.results.values())
    assert all("item_wall_s" in r for r in q.results.values())


def test_width_comes_from_the_payload(monkeypatch):
    """A tp test asks for its chips via the payload; Ray gets that width."""
    seen = []
    _run(monkeypatch, _items(1) + _items(1, width=4),
         lambda p, payload, res: (seen.append(res.get("TPU")), {"ok": True})[1],
         max_items=2, idle_exit_s=2)
    assert sorted(seen) == [1, 4]


def test_width_is_capped_at_host_chips(monkeypatch):
    seen = []
    _run(monkeypatch, _items(1, width=8),
         lambda p, payload, res: (seen.append(res.get("TPU")), {"ok": True})[1],
         chips=4, max_items=1, idle_exit_s=2)
    assert seen == [4]


def test_problem_mismatch_is_faulted_not_silently_dropped(monkeypatch):
    q, _ = _run(monkeypatch, _items(2, problem="megablox_gmm"),
                lambda p, payload, res: {"ok": True},
                idle_exit_s=1)
    assert q.results, "mismatched item must still receive a verdict"
    v = next(iter(q.results.values()))
    assert v["gate"] == "judge_fault"
    assert "megablox_gmm" in v["violations"][0]


def test_task_death_leaves_lease_for_requeue(monkeypatch):
    """If the task process dies we must NOT post a verdict: the lease
    expires and the queue hands the item to another worker."""
    def boom(p, payload, res):
        raise RuntimeError("task died")

    q, done = _run(monkeypatch, _items(1), boom, idle_exit_s=1)
    assert q.results == {}
    assert done == 0


def test_idle_exit_returns_when_queue_stays_empty(monkeypatch):
    import time as _t
    t0 = _t.time()
    q, done = _run(monkeypatch, [], lambda p, payload, res: {"ok": True}, idle_exit_s=1)
    assert done == 0
    assert _t.time() - t0 < 10


def test_item_width_parsing():
    assert ray_pool.item_width({"width": 4}) == 4
    assert ray_pool.item_width({"tp": 2}) == 2
    assert ray_pool.item_width({"chips": "4"}) == 4
    assert ray_pool.item_width({}) == 1
    assert ray_pool.item_width({"width": "junk"}, default=1) == 1


def test_cpu_per_task_leaves_host_headroom():
    assert ray_pool.cpu_per_task() >= 2
