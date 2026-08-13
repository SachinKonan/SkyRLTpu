"""Compile-warm deadline (phase-5 prerequisite).

Phase 4's one residual: the per-grade wall budget is checked BETWEEN phases,
so it cannot bound the single un-cancellable operation that actually wedged a
judge — the XLA compile of the candidate's artifact (measured at 569.5 s for
one candidate that used 3.1 s of chip time). Under lease-requeue that is a
fleet-wide poison pill. These tests pin the three properties that close it:

  1. the compile-warm has its OWN deadline, enforced between compile units;
  2. a candidate that blows it is scored 0 at gate `compile_budget` and
     marked TERMINAL, and the queue never requeues a terminal item;
  3. the same budget bounds the zero-chip CPU pre-gate, so most compile
     bombs never reach a judge at all.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from pallas_arena.judge import worker as worker_mod
from pallas_arena.judge.aot_gate import aot_pregate
from pallas_arena.judge.queue import ArenaQueueClient
from pallas_arena.tests import candidates as cand

ARENA_IMPORT_ROOT = Path(__file__).resolve().parents[2]


def _worker(compile_budget_s, **kw):
    return worker_mod.PersistentWorker(
        "rmsnorm",
        smoke=True,
        cases=["tiny"],
        timing_pairs=4,
        timing_warmup=1,
        correctness_seeds=1,
        export_timeout_s=120.0,
        compile_budget_s=compile_budget_s,
        **kw,
    )


# --------------------------------------------------------------- worker side
def test_compile_budget_rejects_and_is_terminal():
    """A budget of 0 s makes the FIRST between-unit check fire, which is the
    clean path (no process loss). The verdict must be a zero-reward
    `compile_budget` rejection flagged terminal."""
    w = _worker(compile_budget_s=1e-9)
    assert w.boot()["ok"]
    r = w.grade_code(cand.HONEST_RMSNORM, tag="budget-probe")
    assert r["passed"] is False
    assert r["gate"] == "compile_budget", r
    assert r["reward"] == 0.0
    assert r["terminal"] is True
    assert "compile budget" in r["violations"][0]


def test_honest_kernel_survives_the_production_budget():
    """The 90 s production budget must be nowhere near an honest kernel: the
    same candidate that the 0 s budget rejects passes untouched, and its
    compile-warm sits well inside the deadline.

    The bound is DELIBERATELY loose, because the tight one was measuring the
    compute node rather than the kernel. `candidate_compile_s <
    DEFAULT_COMPILE_BUDGET_S / 10` (9 s) measured 5.42 / 8.02 / 13.26 / 24.85 s
    on identical code across four batteries -- a 4.6x swing that tracks node
    load -- and failed 3 of them, which makes the whole suite useless as a
    signal.

    A relative bound against the judge's own boot was tried and was worse: I
    keyed it on `calibration_warm_s`, which is ~0.017 s and is not a compile at
    all, so `20x` of it failed an 8 s compile that the ORIGINAL bound would have
    passed. Same mistake as the kernel measurements this session -- using a
    number without checking what it represents.

    So: one loose absolute bound at half the budget. It clears every observed
    value with margin, and still fails a candidate that is genuinely near the
    90 s deadline, which is the only claim this test needs to make. The tight
    version of this belongs in a benchmark on a quiet node, not in a
    correctness battery that must be green to be worth running.
    """
    w = _worker(compile_budget_s=worker_mod.DEFAULT_COMPILE_BUDGET_S)
    assert w.boot()["ok"]
    r = w.grade_code(cand.HONEST_RMSNORM, tag="honest")
    assert r["passed"], r
    assert r["candidate_compile_s"] < worker_mod.DEFAULT_COMPILE_BUDGET_S / 2, r


def test_compile_budget_verdict_is_cached_so_the_fleet_pays_once():
    """A poison candidate must cost the fleet ONE budget, not one per judge:
    the terminal rejection goes into the hash->reward cache like any other
    verdict, so every later judge serves it for free."""
    import tempfile

    from pallas_arena.judge.cache import RewardCache

    with tempfile.TemporaryDirectory() as td:
        cache = RewardCache(td)
        w = _worker(compile_budget_s=1e-9, cache=cache)
        assert w.boot()["ok"]
        first = w.grade_code(cand.HONEST_RMSNORM, tag="poison")
        assert first["gate"] == "compile_budget"
        second = w.grade_code(cand.HONEST_RMSNORM, tag="poison")
        assert second.get("cache_hit") is True
        assert second["gate"] == "compile_budget" and second["terminal"] is True
        assert second["reward"] == first["reward"] == 0.0


def test_watchdog_fires_inside_a_single_uncancellable_unit():
    """The backstop layer. The between-unit check cannot see an overrun that
    happens INSIDE one compile; the watchdog thread can, because XLA releases
    the GIL while it compiles. A small compile bomb guarantees the overrun
    lands inside the first unit rather than between two. The poster is
    captured here instead of exiting the process, so the mechanism is
    observable."""
    posted = []
    w = _worker(compile_budget_s=0.5)
    w.terminal_poster = posted.append
    assert w.boot()["ok"]
    r = w.grade_code(cand.make_compile_bomb(24), tag="watchdog")
    # whichever layer wins, the verdict is the same
    assert r["gate"] == "compile_budget" and r["terminal"] is True
    assert w.compile_timeouts >= 1, "watchdog thread never observed the overrun"
    assert posted, "watchdog did not post a terminal verdict"
    assert posted[0]["gate"] == "compile_budget" and posted[0]["terminal"] is True


# ------------------------------------------------------------ pre-gate side
def test_pregate_budget_is_enforced_and_bounds_the_wall(capsys):
    """The pre-gate runs the same deadline, enforced by the parent's
    process-group kill on a killable child with no chip anywhere. The
    verdict must be `compile_budget` (not the generic `timeout`) and the
    wall must be bounded BY THE BUDGET rather than by the candidate."""
    t0 = time.perf_counter()
    ok, why = aot_pregate("rmsnorm", cand.COMPILE_BOMB_RMSNORM, smoke=False, compile_budget_s=2.0)
    wall = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[measure] pre-gate budget=2s: {wall:.1f}s -> ok={ok} {why[:70]}")
    assert not ok
    assert "compile_budget" in why, why
    assert wall < 20.0, f"pre-gate took {wall:.1f}s for a 2s budget"


def test_pregate_kills_a_graph_size_bomb_at_zero_chip_cost(capsys):
    """The attack sub-class the pre-gate CAN see: cost carried by graph size,
    so tracing and lowering are expensive on any backend. This one never
    reaches a judge at all."""
    t0 = time.perf_counter()
    ok, why = aot_pregate("rmsnorm", cand.make_compile_bomb(4000), smoke=False, compile_budget_s=20.0)
    wall = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[measure] pre-gate graph-size bomb (depth 4000, budget 20s): {wall:.1f}s -> ok={ok}")
    assert not ok
    assert "compile_budget" in why, why
    assert wall < 60.0


def test_measure_cpu_pregate_cannot_see_a_backend_compile_bomb(capsys):
    """Measurement + documented limit, deliberately not a gate.

    Phase 4 measured a SIX-deep sin chain at 569.5 s inside one XLA:TPU
    compile. The shipped `compile-bomb` is 220 deep -- 37x that graph -- and
    XLA:CPU lowers AND compiles it at production shapes in a couple of
    seconds, because the cost of these bombs lives in the backend's
    fusion/live-range/scheduling passes, which CPU and TPU do not share.

    The conclusion is the one that matters for the fleet: a CPU-only
    pre-gate is NOT a substitute for the judge-side compile deadline. On a
    real client (a v5p training host) the pre-gate child compiles on TPU
    under the same budget and does catch it; on CPU it cannot, by
    construction. If this test ever starts failing because the CPU pre-gate
    began rejecting the bomb, that is good news, not a regression."""
    t0 = time.perf_counter()
    ok_bomb, _ = aot_pregate("rmsnorm", cand.COMPILE_BOMB_RMSNORM, smoke=False, compile_budget_s=90.0)
    bomb_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    ok_honest, _ = aot_pregate("rmsnorm", cand.HONEST_RMSNORM, smoke=False, compile_budget_s=90.0)
    honest_s = time.perf_counter() - t0
    with capsys.disabled():
        print(
            f"\n[measure] CPU pre-gate at production shapes, 90s budget: "
            f"compile-bomb(220)={bomb_s:.1f}s ok={ok_bomb} | honest={honest_s:.1f}s ok={ok_honest}"
        )
    assert ok_honest, "the honest control must always pass"


@pytest.mark.parametrize("depth", [24, 220])
def test_measure_compile_bomb_export_cost(depth, capsys):
    """Measurement, not a gate. The bomb has to reach the JUDGE for the
    compile-warm deadline to be the thing that catches it, so its cost in
    the export child (bounded separately at 240 s) must stay small: all its
    weight belongs in the backend compile, not in tracing. Printed so the
    phase report can quote real numbers for the depth that shipped."""
    from pallas_arena.judge import grader

    w = _worker(compile_budget_s=None)
    assert w.boot()["ok"]
    sigs, *_ = w._signatures()
    t0 = time.perf_counter()
    child = grader.grade(
        "rmsnorm",
        cand.make_compile_bomb(depth),
        mode="aot_export",
        smoke=True,
        timeout_s=240.0,
        cache=None,
        export_signatures=sigs,
        export_platforms=[w.platform],
    )
    wall = time.perf_counter() - t0
    with capsys.disabled():
        print(
            f"\n[measure] compile-bomb depth={depth}: export child wall={wall:.1f}s "
            f"export_s={child.get('export_s')} passed={child.get('passed')} gate={child.get('gate')}"
        )
    assert wall < 240.0


def test_pregate_still_accepts_honest_kernels_under_the_budget(capsys):
    """The control for the test above: same budget, same production shapes,
    an honest kernel sails through. Without this the bomb test would also
    pass if the pre-gate simply rejected everything."""
    t0 = time.perf_counter()
    ok, why = aot_pregate("rmsnorm", cand.HONEST_RMSNORM, smoke=False, compile_budget_s=90.0)
    wall = time.perf_counter() - t0
    with capsys.disabled():
        print(f"\n[measure] pre-gate honest (production shapes): {wall:.1f}s -> ok={ok}")
    assert ok, why


def test_pregate_lowers_the_gradient_functional():
    """The 569.5 s compile lived in the GRAD graph, so a fwd-only pre-gate
    waves that whole attack class through. This candidate traces and lowers
    perfectly forward and is undifferentiable in reverse mode — it must be
    rejected, which is only possible if the pre-gate lowers the gradient."""
    bad_grad = """
import jax
import jax.numpy as jnp

def kernel(x, g):
    x32 = x.astype(jnp.float32)
    y = x32 * jax.lax.rsqrt(jnp.mean(jnp.square(x32), -1, keepdims=True) + 1e-6) * g
    # while_loop has no reverse-mode rule: fine forward, fatal under jax.grad
    _, y = jax.lax.while_loop(lambda c: c[0] < 2, lambda c: (c[0] + 1, c[1] * 1.0), (0, y))
    return y
"""
    ok, why = aot_pregate("rmsnorm", bad_grad, smoke=True)
    assert not ok and "pregate" in why, why


# --------------------------------------------------------------- queue side
def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _start_queue(port: int, lease_timeout_s: float) -> subprocess.Popen:
    import os

    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ARENA_IMPORT_ROOT), env.get("PYTHONPATH", "")])
    proc = subprocess.Popen(
        [sys.executable, "-m", "pallas_arena.judge.queue", "--port", str(port), "--lease-timeout", str(lease_timeout_s)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + 30
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/status", timeout=2) as r:
                if json.loads(r.read()).get("ok"):
                    return proc
        except Exception:
            time.sleep(0.3)
    proc.kill()
    raise RuntimeError("queue never became healthy")


def _post(port, path, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return json.loads(r.read())


def _get(port, path):
    with urllib.request.urlopen(f"http://127.0.0.1:{port}{path}", timeout=10) as r:
        if r.status == 204:
            return None
        return json.loads(r.read())


@pytest.mark.parametrize("lease_timeout_s", [1.0])
def test_terminal_result_is_never_requeued(lease_timeout_s):
    """The whole point of TERMINAL: a poison candidate must not migrate. A
    normal rejection would also be `done` here, so the test proves the
    stronger property — the item is flagged terminal, counted, and the
    expiry sweep skips it even after the lease is long gone."""
    port = _free_port()
    q = _start_queue(port, lease_timeout_s)
    try:
        client = ArenaQueueClient(f"http://127.0.0.1:{port}")
        wid = client.submit("rmsnorm", cand.COMPILE_BOMB_RMSNORM, tag="compile-bomb#s0")
        lease = _get(port, "/work?worker_id=judge-1")
        assert lease["work_id"] == wid
        _post(
            port,
            "/result",
            {
                "lease_id": lease["lease_id"],
                "work_id": wid,
                "terminal": True,
                "result": {"passed": False, "gate": "compile_budget", "reward": 0.0, "terminal": True},
            },
        )
        time.sleep(lease_timeout_s + 0.5)  # past any lease expiry
        for _ in range(3):
            assert _get(port, "/work?worker_id=judge-2") is None, "terminal item was handed to a second judge"
        got = _get(port, f"/result/{wid}")
        assert got["done"] and got["terminal"] is True and got["attempts"] == 1
        st = _get(port, "/status")
        assert st["terminal_rejections"] == 1
        assert st["requeues"] == 0 and st["queue_depth"] == 0
    finally:
        q.kill()
        q.wait(timeout=5)
