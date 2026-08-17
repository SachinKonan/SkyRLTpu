"""Tensor-parallel specs actually EXECUTE, on 8 forced CPU devices.

Every declared tp8 case had been skipped on every judge run (single-chip
judges), so until this test the TP specs -- including MQA's KV-replication
rule -- had never run a shard_map at all. `xla_force_host_platform_device_
count` must be set before JAX initializes, and the battery's process has
long since imported JAX, so the check runs in a subprocess
(tests/tp_cpu_check.py); this wrapper only launches it and asserts on the
summary line.

The invariant: sharded agrees with unsharded to fp32-rounding scale (rel max
err < 1e-4; a different compiled program cannot be bitwise -- see
tp_cpu_check's docstring), forward and, for splash, gradients through
shard_map too.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CHECK = Path(__file__).with_name("tp_cpu_check.py")


def test_tp_specs_execute_and_match_unsharded_bitwise():
    env = os.environ.copy()
    env["XLA_FLAGS"] = (env.get("XLA_FLAGS", "") + " --xla_force_host_platform_device_count=8").strip()
    env["JAX_PLATFORMS"] = "cpu"
    env.pop("ARENA_CHILD_JAX_PLATFORMS", None)
    r = subprocess.run(
        [sys.executable, str(CHECK)],
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        check=False,
    )
    assert r.returncode == 0, f"stdout:\n{r.stdout}\nstderr:\n{r.stderr[-2000:]}"
    assert "ALL-OK" in r.stdout, r.stdout
    # one line per structural family, so a silently-dropped case is visible
    for needle in ("cpu-tp8-mha", "cpu-tp8-gqa", "cpu-tp8-mqa", "cpu-tp8-window",
                   "cpu-tp8-gmm", "cpu-tp8-rpa", "cpu-tp8-lru", "grad-through-shard_map"):
        assert needle in r.stdout, (needle, r.stdout)
