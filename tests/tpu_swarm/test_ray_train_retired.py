import os
from types import SimpleNamespace

import pytest
from dataclasses import replace

pytest.importorskip("psutil")
from tpu.swarm.ray_train import retired
from tpu.swarm.ray_train.config import Config


TASK = "sky-managed-2026-09-05-19-47-24-615848_meta-wt16-carry-g0-muse_271-0"


class FakeProcess:
    def __init__(self, pid, args, task=TASK, children=()):
        self.pid, self.task, self.child_processes = pid, task, children
        self.info = {"pid": pid, "cmdline": args}
        self.signals = []

    def status(self):
        return "running"

    def environ(self):
        return {"SKYPILOT_TASK_ID": self.task} if self.task else {}

    def create_time(self):
        return 12345.0

    def cmdline(self):
        return self.info["cmdline"]

    def uids(self):
        return SimpleNamespace(real=os.getuid())

    def children(self, recursive):
        assert recursive
        return self.child_processes

    def terminate(self):
        self.signals.append("TERM")

    def kill(self):
        self.signals.append("KILL")


def install_processes(monkeypatch, processes):
    monkeypatch.setattr(retired.psutil, "process_iter", lambda attrs: processes)
    monkeypatch.setattr(retired, "wait_for_exit", lambda processes, timeout: [])


def test_retirement_stops_only_exact_task_roots_and_descendants(monkeypatch):
    engine = FakeProcess(2, ["VLLM::EngineCore"])
    root = FakeProcess(1, ["python", "/home/user/vllm_tpu_server.py"], children=[engine])
    sky = FakeProcess(3, ["raylet", "--port=6380"])
    unrelated = FakeProcess(4, ["python", "analysis.py"])
    install_processes(monkeypatch, [root, engine, sky, unrelated])
    assert retired.retire_workloads([TASK]) == [1, 2]
    assert root.signals == engine.signals == ["TERM"]
    assert not sky.signals and not unrelated.signals


@pytest.mark.parametrize("task", [None, TASK + "-new"])
def test_unknown_workload_prevents_all_cleanup(monkeypatch, task):
    old = FakeProcess(1, ["python", "-m", "skyrl.tinker.api"])
    active = FakeProcess(2, ["python", "/home/user/vllm_tpu_server.py"], task=task)
    install_processes(monkeypatch, [old, active])
    with pytest.raises(RuntimeError, match="not explicitly retired"):
        retired.retire_workloads([TASK])
    assert not old.signals and not active.signals


def test_retirement_rejects_broad_task_selectors():
    raw = Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32.json").to_dict()
    raw["retired_task_ids"] = ["sky-managed-*"]
    with pytest.raises(ValueError, match="exact managed task IDs"):
        Config.from_dict(raw)


def test_retired_cell_supervisor_is_recognized_but_not_tmux_server():
    assert retired.workload_kind(["bash", "/home/user/cell_worker.sh"]) == "cell supervisor"
    assert retired.workload_kind(["bash", "/home/user/sidecar_old.sh"]) == "cell supervisor"
    assert retired.workload_kind(["tmux: server"]) is None


def test_orphan_engine_requires_exact_retired_task(monkeypatch):
    engine = FakeProcess(7, ["VLLM::EngineCore", ""])
    install_processes(monkeypatch, [engine])
    assert retired.retire_workloads([TASK]) == [7]
    assert engine.signals == ["TERM"]


def test_unknown_orphan_engine_prevents_all_cleanup(monkeypatch):
    root = FakeProcess(1, ["python", "/home/user/vllm_tpu_server.py"])
    engine = FakeProcess(7, ["VLLM::EngineCore"], task=None)
    install_processes(monkeypatch, [root, engine])
    with pytest.raises(RuntimeError, match="not explicitly retired"):
        retired.retire_workloads([TASK])
    assert not root.signals and not engine.signals


def test_engine_without_task_is_covered_by_audited_parent(monkeypatch):
    engine = FakeProcess(7, ["VLLM::EngineCore"], task=None)
    root = FakeProcess(1, ["python", "/home/user/vllm_tpu_server.py"], children=[engine])
    install_processes(monkeypatch, [root, engine])
    assert retired.retire_workloads([TASK]) == [1, 7]


def test_bootstrap_retires_before_checking_ports(monkeypatch, tmp_path):
    from tpu.swarm.ray_train import bootstrap
    config = replace(Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32.json"), retired_task_ids=[TASK])
    calls = []
    monkeypatch.setattr(retired, "retire_workloads", lambda *args: calls.append("retire") or [])
    monkeypatch.setattr(bootstrap, "check_ports_available", lambda ports: calls.append("ports"))
    bootstrap.retire_before_port_check(config, "10.130.0.244", tmp_path / "bootstrap.jsonl")
    assert calls == ["retire", "ports"]


def test_bootstrap_refuses_unknown_workload_before_port_check(monkeypatch, tmp_path):
    from tpu.swarm.ray_train import bootstrap
    config = replace(Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32.json"), retired_task_ids=[TASK])
    calls = []
    def refuse(*args):
        raise RuntimeError("existing inference is not explicitly retired")
    monkeypatch.setattr(retired, "retire_workloads", refuse)
    monkeypatch.setattr(bootstrap, "check_ports_available", lambda ports: calls.append("ports"))
    with pytest.raises(RuntimeError, match="not explicitly retired"):
        bootstrap.retire_before_port_check(config, "10.130.0.244", tmp_path / "bootstrap.jsonl")
    assert not calls


def test_retirement_wait_does_not_require_pidfd(monkeypatch):
    import subprocess
    import sys
    def unsupported(*args, **kwargs):
        raise OSError(22, "Invalid argument")
    monkeypatch.setattr(os, "pidfd_open", unsupported, raising=False)
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    process = retired.psutil.Process(child.pid)
    try:
        assert retired.wait_for_exit([process], timeout=0) == [process]
        child.terminate()
        assert retired.wait_for_exit([process], timeout=5) == []
    finally:
        if child.poll() is None:
            child.kill()
        child.wait(timeout=5)


@pytest.mark.parametrize("mismatch", [None, "boot_id", "created", "command_sha256"])
def test_legacy_process_requires_exact_audited_identity(monkeypatch, mismatch):
    process = FakeProcess(9, ["python", "/home/user/vllm_tpu_server.py"], task=None)
    install_processes(monkeypatch, [process])
    boot_id = retired.Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    audit = {"boot_id": boot_id, "processes": [retired.process_identity(process)]}
    if mismatch == "boot_id":
        audit["boot_id"] = "different boot"
    elif mismatch:
        audit["processes"][0][mismatch] = "changed"
    if mismatch:
        with pytest.raises(RuntimeError):
            retired.retire_workloads([], audit)
        assert not process.signals
    else:
        assert retired.retire_workloads([], audit) == [9]
        assert process.signals == ["TERM"]
