"""CPU-only controller tests; no Ray cluster or cloud resources are started."""
from types import SimpleNamespace

import pytest

pytest.importorskip("ray.serve")
from tpu.swarm.ray_train import controller as module
from tpu.swarm.ray_train.config import Config


def controller(tmp_path):
    cfg = Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32.json").to_dict()
    cfg["root"] = str(tmp_path)
    instance = module.Controller(Config.from_dict(cfg), [f"10.0.0.{i}" for i in range(4)])
    instance.events = []
    instance.report = lambda event, **fields: instance.events.append((event, fields))
    instance.prepared = {"ready": True}
    instance.calls = []

    def remote(rank, kind):
        def call():
            ref = (rank, kind, len(instance.calls))
            instance.calls.append(ref)
            return ref
        return SimpleNamespace(remote=call)

    instance.hosts = [SimpleNamespace(stop=remote(r, "stop"), sync_compile=remote(r, "compile"),
                                     sync_run=remote(r, "run"), stop_client=remote(r, "stop_client")) for r in range(8)]
    return instance


def test_slow_host_and_run_upload_do_not_block_peer_sync(tmp_path, monkeypatch):
    control = controller(tmp_path)
    now = [1000]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.ray, "wait", lambda refs, **kw: (
        [r for r in refs if r[0] != 0], [r for r in refs if r[0] == 0]))
    monkeypatch.setattr(module.ray, "get", lambda ref: 2)
    control.writeback_tick()
    assert len(control.calls) == 9
    now[0] += control.config.cache.sync_seconds
    control.writeback_tick()
    assert len(control.calls) == 16
    assert len([r for r in control.calls if r[0] == 0]) == 2
    assert all(len([r for r in control.calls if r[0] == rank]) == 2 for rank in range(1, 8))


def test_failed_sync_retries_on_next_interval(tmp_path, monkeypatch):
    control = controller(tmp_path)
    now = [1000]
    monkeypatch.setattr(module.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(module.ray, "wait", lambda refs, **kw: (refs, []))

    def get(ref):
        if ref[0] == 3:
            raise RuntimeError("GCS unavailable")
        return 1

    monkeypatch.setattr(module.ray, "get", get)
    control.writeback_tick()
    control.writeback_tick()
    assert len(control.calls) == 9
    assert any(event == "writeback_retry_pending" and fields["rank"] == 3
               for event, fields in control.events)
    now[0] += control.config.cache.sync_seconds
    control.writeback_tick()
    assert len(control.calls) == 18
    control.stopping.set()
    control.writeback_tick()
    assert len(control.calls) == 18


def test_final_flush_reaches_healthy_hosts_after_peer_failure(tmp_path, monkeypatch):
    control = controller(tmp_path)
    monkeypatch.setattr(module.serve, "shutdown", lambda: None)
    monkeypatch.setattr(module.ray, "wait", lambda refs, **kw: (refs[:1], refs[1:]))

    def get(ref):
        if ref[0] == 2:
            raise RuntimeError("host lost")
        return 1

    monkeypatch.setattr(module.ray, "get", get)
    control.close()
    assert len([r for r in control.calls if r[1] == "compile"]) == 8
    assert len([r for r in control.calls if r[1] == "run"]) == 1
    completed = [fields["rank"] for event, fields in control.events
                 if event == "final_compile_writeback_complete"]
    assert completed == [0, 1, 3, 4, 5, 6, 7]
    assert any(event == "final_compile_writeback_error" and fields["rank"] == 2
               for event, fields in control.events)
