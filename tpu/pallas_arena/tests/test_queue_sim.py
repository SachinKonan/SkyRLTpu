"""Phase-3 local simulation: real queue process + 5 real worker processes
(mock-timed grading) with chaos — worker kills mid-lease, queue restart —
and exact lease/requeue accounting. Runs under sbatch with the battery."""

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

from pallas_arena.judge.queue import ArenaQueueClient

ARENA_IMPORT_ROOT = Path(__file__).resolve().parents[2]


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def _env():
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([str(ARENA_IMPORT_ROOT), env.get("PYTHONPATH", "")])
    env["JAX_PLATFORMS"] = "cpu"
    return env


def _start_queue(port: int, lease_timeout_s: float) -> subprocess.Popen:
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pallas_arena.judge.queue",
            "--port",
            str(port),
            "--lease-timeout",
            str(lease_timeout_s),
        ],
        env=_env(),
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


def _start_worker(port: int, worker_id: str, grade_s: float) -> subprocess.Popen:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "pallas_arena.judge.worker",
            "--problem",
            "rmsnorm",
            "--queue",
            f"http://127.0.0.1:{port}",
            "--sim-mode",
            "mock",
            "--mock-grade-s",
            str(grade_s),
            "--worker-id",
            worker_id,
            "--poll-s",
            "0.2",
        ],
        env=_env(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _kill_all(procs):
    for p in procs:
        try:
            p.kill()
        except Exception:
            pass
    for p in procs:
        try:
            p.wait(timeout=5)
        except Exception:
            pass


def test_five_workers_drain_queue_with_exact_accounting():
    port = _free_port()
    procs = [_start_queue(port, lease_timeout_s=20.0)]
    try:
        client = ArenaQueueClient(f"http://127.0.0.1:{port}")
        n = 25
        ids = [client.submit("rmsnorm", f"# item {i}\ndef kernel(): pass") for i in range(n)]
        procs += [_start_worker(port, f"sim-{i}", grade_s=0.4) for i in range(5)]
        results = client.wait(ids, timeout_s=90.0, poll_s=0.5)
        assert len(results) == n, f"only {len(results)}/{n} completed"
        assert all(r["mock"] for r in results.values())
        # deterministic mock rewards: same code -> same reward
        r_dup = client.submit("rmsnorm", "# item 0\ndef kernel(): pass")
        got = client.wait([r_dup], timeout_s=30.0)[r_dup]
        assert got["reward"] == results[ids[0]]["reward"]
        s = client.status()
        assert s["completed"] >= n
        assert s["queue_depth"] == 0
        # all five workers actually pulled work
        assert len(s["workers_seen"]) == 5
    finally:
        _kill_all(procs)


def test_chaos_kill_workers_mid_lease_requeues_and_completes():
    """Two slow workers (30s grades, heartbeating) grab items and are
    SIGKILLed mid-grade: their leases expire (heartbeats stop) and the
    items complete on the surviving fast workers."""
    port = _free_port()
    lease = 4.0
    procs = [_start_queue(port, lease_timeout_s=lease)]
    slow, fast = [], []
    try:
        client = ArenaQueueClient(f"http://127.0.0.1:{port}")
        # slow workers first so they grab the head items
        slow = [_start_worker(port, f"slow-{i}", grade_s=30.0) for i in range(2)]
        n = 16
        ids = [client.submit("rmsnorm", f"# chaos {i}\n") for i in range(n)]
        time.sleep(2.0)  # slow workers now hold leases mid-grade
        st = client.status()
        assert st["leased"] >= 1, "slow workers should be holding leases"
        for p in slow:
            os.kill(p.pid, signal.SIGKILL)
        fast = [_start_worker(port, f"fast-{i}", grade_s=0.3) for i in range(3)]
        results = client.wait(ids, timeout_s=120.0, poll_s=0.5)
        assert len(results) == n, f"only {len(results)}/{n} after chaos"
        s = client.status()
        assert s["requeues"] >= 2, s  # the two killed leases came back
        assert s["completed"] >= n
    finally:
        _kill_all(procs + slow + fast)


def test_chaos_queue_restart_client_resubmits():
    """The queue is memoryless by design: kill it mid-run, restart on the
    same port, and the CLIENT (source of truth) resubmits what it never saw
    complete. Every original work id ends with a result."""
    port = _free_port()
    queue_proc = _start_queue(port, lease_timeout_s=10.0)
    workers = []
    try:
        client = ArenaQueueClient(f"http://127.0.0.1:{port}")
        n = 12
        ids = [client.submit("rmsnorm", f"# restart {i}\n") for i in range(n)]
        workers = [_start_worker(port, f"rw-{i}", grade_s=0.5) for i in range(2)]
        # let roughly half complete, then murder the queue
        deadline = time.time() + 30
        while time.time() < deadline:
            if client.status()["completed"] >= n // 2:
                break
            time.sleep(0.3)
        os.kill(queue_proc.pid, signal.SIGKILL)
        queue_proc.wait(timeout=5)
        time.sleep(1.0)
        for attempt in range(5):
            try:
                queue_proc = _start_queue(port, lease_timeout_s=10.0)
                break
            except RuntimeError:
                if attempt == 4:
                    raise
                time.sleep(2.0)
        results = client.wait(ids, timeout_s=120.0, poll_s=0.5)
        assert len(results) == n, f"only {len(results)}/{n} after queue restart"
    finally:
        _kill_all([queue_proc] + workers)


@pytest.mark.parametrize("n_items", [6])
def test_worker_max_items_exits_cleanly(n_items):
    port = _free_port()
    procs = [_start_queue(port, lease_timeout_s=10.0)]
    try:
        client = ArenaQueueClient(f"http://127.0.0.1:{port}")
        ids = [client.submit("rmsnorm", f"# fin {i}\n") for i in range(n_items)]
        w = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "pallas_arena.judge.worker",
                "--problem",
                "rmsnorm",
                "--queue",
                f"http://127.0.0.1:{port}",
                "--sim-mode",
                "mock",
                "--mock-grade-s",
                "0.1",
                "--max-items",
                str(n_items),
                "--poll-s",
                "0.2",
            ],
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        procs.append(w)
        out, _ = w.communicate(timeout=60)
        assert w.returncode == 0, out.decode()[-500:]
        assert f"done after {n_items} items" in out.decode()
        assert len(client.wait(ids, timeout_s=10.0)) == n_items
    finally:
        _kill_all(procs)
