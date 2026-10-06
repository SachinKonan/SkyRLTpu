from __future__ import annotations

import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[2]


def test_cell_lifecycle_uses_exact_tmux_targets():
    launch = (REPO / "tpu/launch_cell.sh").read_text()
    monitor = (REPO / "tpu/jobman/cell_monitor.sh").read_text()

    assert 'tmux kill-session -t "=${SESSION}-backup"' in launch
    assert 'tmux kill-session -t "=$SESSION"' in launch
    assert 'tmux has-session -t "=$SESSION"' in launch
    assert '-t "${SESSION}-backup"' not in launch
    assert '-t "$SESSION"' not in launch

    exact_client_target = (
        '-t "=$SESSION"' in monitor or "-t =cell" in monitor
    )
    assert exact_client_target
    if "${SESSION}-backup" in monitor or "cell-backup" in monitor:
        assert (
            '-t "=${SESSION}-backup"' in monitor or "-t =cell-backup" in monitor
        )
    assert '-t "${SESSION}-backup"' not in monitor
    assert '-t "$SESSION"' not in monitor
    assert "-t cell" not in monitor
    assert "-t cell-backup" not in monitor


def test_engine_lifecycle_uses_exact_tmux_targets():
    worker = (REPO / "tpu/jobman/cell_worker.sh").read_text()
    colocated = (REPO / "tpu/start_colocated_vllm_tinker.sh").read_text()
    vllm = (REPO / "tpu/start_vllm_tpu.sh").read_text()
    reconcile = (REPO / "tpu/swarm/reconcile_v4_64_host_role.sh").read_text()

    assert "tmux kill-session -t skyrl-tinker" not in worker
    assert "tmux kill-session -t skyrl-vllm" not in worker
    assert "tmux kill-session -t skyrl-tinker" not in colocated
    assert "tmux has-session -t vllm-tpu" not in colocated
    assert "tmux kill-session -t vllm-tpu" not in vllm
    assert "tmux kill-session -t vllm-tpu" not in reconcile
    assert "tmux kill-session -t skyrl-tinker" not in reconcile
    assert 'tmux kill-session -t "$session"' not in reconcile


@pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")
def test_exact_client_target_does_not_match_backup_session():
    socket_name = f"skyrl-exact-target-{os.getpid()}-{uuid.uuid4().hex}"
    tmux = ["tmux", "-L", socket_name]
    subprocess.run(
        [*tmux, "new-session", "-d", "-s", "cell-backup", "sleep", "60"],
        check=True,
    )
    try:
        assert subprocess.run(
            [*tmux, "has-session", "-t", "=cell"], check=False
        ).returncode != 0
        assert subprocess.run(
            [*tmux, "kill-session", "-t", "=cell"], check=False
        ).returncode != 0
        subprocess.run([*tmux, "has-session", "-t", "=cell-backup"], check=True)
    finally:
        subprocess.run([*tmux, "kill-server"], check=False)
