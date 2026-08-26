"""Control-flow tests for the per-test Ray grading pool (fake Ray + fake
queue) and pure-math tests for the collector.

The pool's first real execution is unattended on a judge host, so everything
that does not need TPU -- stage-0 short-circuit, per-test fan-out, width
from the case name, sibling cancellation on candidate-fault, judge-fault
exclusion, lease requeue on task death, idle-exit, and the reward fold --
is exercised here on CPU.
"""

from __future__ import annotations

import sys
import types

import pytest

from pallas_arena.judge import collect, ray_pool


# --------------------------------------------------------------- collector
def _ok_result(case, score=1.5, floor=0.05, grad=None, grad_ok=True):
    r = {
        "passed": True, "gate": "all", "score": score,
        "task_noise_floor": floor, "task_boot_s": 3.0,
        "latencies": {case: {"ref_median_s": 1e-3, "cand_median_s": 1e-3 / score}},
        "grad_ok": grad_ok, "grad_scores": ({case: grad} if grad is not None else {}),
    }
    return {"result": r}


def test_collect_geomean_and_bwd_fold():
    entries = {
        "probe-a": _ok_result("probe-a", score=2.0, grad=2.0),
        "probe-b": _ok_result("probe-b", score=0.5, grad=0.5),
    }
    m = collect.merge_case_results("rg_lru", entries)
    assert m["passed"]
    assert abs(m["score"] - 1.0) < 1e-9          # geomean(2, .5) = 1
    # bwd factors [2.0, 0.5] fold with the fwd factors
    assert abs(m["reward_with_bwd"] - 1.0) < 1e-9
    assert m["n_scored_cases"] == 2 and m["n_bwd_factors"] == 2


def test_collect_correct_everywhere_zeroes_on_any_candidate_fault():
    entries = {
        "probe-a": _ok_result("probe-a", score=3.0),
        "probe-b": {"result": {"passed": False, "gate": "correctness",
                               "violations": ["max err 2.8 exceeds tolerance"],
                               "task_noise_floor": 0.05}},
    }
    m = collect.merge_case_results("rg_lru", entries)
    assert not m["passed"]
    assert m["reward_with_bwd"] == 0.0
    assert "probe-b" in m["violations"][0]


def test_collect_judge_fault_excludes_not_zeroes():
    entries = {
        "probe-a": _ok_result("probe-a", score=2.0, grad=2.0),
        "tp4-x": {"judge_fault": "tp control failed: 1.31"},
    }
    m = collect.merge_case_results("rg_lru", entries)
    assert m["passed"]
    assert m["excluded_cases"] == {"tp4-x": "tp control failed: 1.31"}
    assert abs(m["score"] - 2.0) < 1e-9


def test_collect_absent_bwd_floors_never_beats_slow_correct():
    slow = collect.merge_case_results(
        "rg_lru", {"probe-a": _ok_result("probe-a", score=2.0, grad=0.01)})
    absent = collect.merge_case_results(
        "rg_lru", {"probe-a": _ok_result("probe-a", score=2.0, grad=None, grad_ok=False)})
    assert absent["reward_with_bwd"] <= slow["reward_with_bwd"] + 1e-12


def test_collect_noise_floor_collapses_ties():
    m = collect.merge_case_results(
        "rg_lru", {"probe-a": _ok_result("probe-a", score=1.03, floor=0.05)})
    assert m["reward"] == 1.0                     # inside the floor -> exactly 1.0


def test_case_width_and_holdout_convention():
    assert collect.case_width("tp4-4x2048x2560") == 4
    assert collect.case_width("tp8-h32-s4096") == 8
    assert collect.case_width("probe-2x1024x2560") == 1
    assert collect.is_holdout("tp4-holdout-2x1500x2560")
    assert not collect.is_holdout("probe-2x1024x2560")


def test_classify_task_error():
    assert collect.classify_task_error("RuntimeUnexpectedCoreHalt: boom") == "fatal"
    assert collect.classify_task_error("WorkerCrashedError: oom") == "judge"


# ------------------------------------------------------------ pool control
class _FakeRef:
    _n = 0

    def __init__(self, fn):
        self._fn = fn
        _FakeRef._n += 1
        self._id = _FakeRef._n

    def __hash__(self):
        return self._id

    def __eq__(self, other):
        return self is other


def _install_fake_ray(monkeypatch, grade_impl):
    class FakeRemoteFn:
        def __init__(self, resources=None):
            self.resources = resources or {}

        def options(self, resources=None, **kw):
            return FakeRemoteFn(resources)

        def remote(self, problem, case, payload, cfg):
            res = self.resources
            return _FakeRef(lambda: grade_impl(problem, case, payload, res))

    fake = types.ModuleType("ray")
    fake.init = lambda **kw: None
    fake.remote = lambda **kw: (lambda fn: FakeRemoteFn())
    fake.get = lambda ref: ref._fn()
    fake.wait = lambda refs, num_returns=1, timeout=None: (refs[:num_returns], refs[num_returns:])
    fake.cancel = lambda ref, force=False: None
    monkeypatch.setitem(sys.modules, "ray", fake)
    return fake


class _FakeQueue:
    def __init__(self, items):
        self.pending = list(items)
        self.results = {}

    def http(self, url, payload=None, timeout=30.0):
        if "/work" in url:
            if not self.pending:
                return None
            it = self.pending.pop(0)
            return {"work_id": it["work_id"], "lease_id": "lease-" + it["work_id"],
                    "lease_timeout_s": 60.0, "attempt": 1, "payload": it["payload"]}
        if "/heartbeat" in url:
            return {}
        if "/result" in url:
            self.results[payload["work_id"]] = payload["result"]
            return {}
        raise AssertionError(f"unexpected url {url}")


CASES = {"rg_lru": ["probe-a", "probe-b", "tp4-c", "tp8-d"]}


def _items(n, problem="rg_lru"):
    return [{"work_id": f"w{i}", "payload": {"problem": problem, "code": f"# {i}"}}
            for i in range(n)]


def _run(monkeypatch, items, grade_impl, *, pregate=None, chips=4, **kw):
    q = _FakeQueue(items)
    _install_fake_ray(monkeypatch, grade_impl)
    monkeypatch.setattr(ray_pool, "_http_json", q.http)
    monkeypatch.setattr(ray_pool, "stage0_pregate",
                        pregate or (lambda p, c: {"passed": True, "gate": "all"}))
    cfg = {"cases_by_problem": CASES}
    done = ray_pool.run_pool("http://fake", ["rg_lru"], chips=chips, cfg=cfg,
                             poll_s=0.01, **kw)
    return q, done


def _grade_ok(problem, case, payload, res):
    return {"passed": True, "gate": "all", "score": 2.0, "grad_ok": True,
            "grad_scores": {case: 2.0}, "task_noise_floor": 0.05}


def test_per_test_fanout_and_width(monkeypatch):
    seen = {}
    def impl(problem, case, payload, res):
        seen[case] = res.get("TPU")
        return _grade_ok(problem, case, payload, res)

    q, done = _run(monkeypatch, _items(1), impl, max_items=1, idle_exit_s=2)
    assert done == 1
    # tp8 is skipped (max width 4); the rest ran at their declared widths
    assert seen == {"probe-a": 1, "probe-b": 1, "tp4-c": 4}
    v = q.results["w0"]
    assert v["passed"] and v["skipped_cases"].get("tp8-d")
    assert v["n_scored_cases"] == 3


def test_stage0_pregate_short_circuits_without_chip_tasks(monkeypatch):
    calls = []
    def impl(problem, case, payload, res):
        calls.append(case)
        return _grade_ok(problem, case, payload, res)

    q, done = _run(monkeypatch, _items(1), impl,
                   pregate=lambda p, c: {"passed": False, "gate": "pregate",
                                         "violations": ["SyntaxError: bad"]},
                   max_items=1, idle_exit_s=2)
    assert done == 1
    assert calls == []                       # no chip task ever ran
    v = q.results["w0"]
    assert not v["passed"] and v["gate"] == "pregate"


def test_candidate_fault_cancels_siblings(monkeypatch):
    def impl(problem, case, payload, res):
        if case == "probe-a":
            return {"passed": False, "gate": "correctness",
                    "violations": ["wrong"], "task_noise_floor": 0.05}
        return _grade_ok(problem, case, payload, res)

    q, done = _run(monkeypatch, _items(1), impl, max_items=1, idle_exit_s=2)
    v = q.results["w0"]
    assert not v["passed"]
    assert v["reward_with_bwd"] == 0.0
    # siblings that had not finished are recorded as cancelled, not scored
    assert any("cancelled" in str(x) for x in v["excluded_cases"].values())


def test_task_death_with_halt_is_candidate_fatal(monkeypatch):
    def impl(problem, case, payload, res):
        if case == "probe-b":
            raise RuntimeError("RuntimeUnexpectedCoreHalt: device dead")
        return _grade_ok(problem, case, payload, res)

    q, done = _run(monkeypatch, _items(1), impl, max_items=1, idle_exit_s=2)
    v = q.results["w0"]
    assert not v["passed"] and v["gate"] == "runtime_halt"


def test_task_death_generic_is_excluded_not_fatal(monkeypatch):
    def impl(problem, case, payload, res):
        if case == "probe-b":
            raise RuntimeError("worker died: oom")
        return _grade_ok(problem, case, payload, res)

    q, done = _run(monkeypatch, _items(1), impl, max_items=1, idle_exit_s=2)
    v = q.results["w0"]
    assert v["passed"]
    assert "probe-b" in v["excluded_cases"]


def test_idle_exit_returns_when_queue_stays_empty(monkeypatch):
    import time as _t
    t0 = _t.time()
    q, done = _run(monkeypatch, [], _grade_ok, idle_exit_s=1)
    assert done == 0 and _t.time() - t0 < 10


def test_cpu_per_task_leaves_host_headroom():
    assert ray_pool.cpu_per_task() >= 2
