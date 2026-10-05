import base64
from dataclasses import replace
import hashlib
import json
import os
from types import SimpleNamespace

import pytest

from tpu.swarm.ray_train import checkpoint_retention as retention
from tpu.swarm.ray_train.cache import Object
from tpu.swarm.ray_train.config import Config


@pytest.fixture
def owned(tmp_path, monkeypatch):
    monkeypatch.setenv("SKYPILOT_TASK_ID", "sky-managed-test_old_370-0")
    monkeypatch.setattr(retention, "run_is_active", lambda *args: False)
    config = Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32_budget.json")
    config = replace(config, run_id="old", root=str(tmp_path))
    fd = retention.acquire_run_lease(config, tmp_path)
    os.close(fd)
    run = tmp_path / "runs/old"
    path = run / "checkpoints/model_test/000001.tar.gz"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"checkpoint contents")
    obj = Object(config.run_gcs + "/checkpoints/model_test/000001.tar.gz",
                 "model_test/000001.tar.gz", path.stat().st_size, "123",
                 base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode(), None)
    calls = []

    def listing(prefix, allow_empty):
        calls.append(prefix)
        assert prefix == config.run_gcs + "/checkpoints" and allow_empty
        return [obj]

    gcs = SimpleNamespace(list=listing)
    return SimpleNamespace(root=tmp_path, config=config, run=run, path=path, obj=obj,
                           gcs=gcs, calls=calls, log=tmp_path / "cleanup.jsonl")


def clean(owned, **kwargs):
    return retention.reclaim_checkpoints(owned.root, "current", owned.gcs, owned.log, **kwargs)


def test_reclaims_only_verified_archives_and_keeps_state(owned):
    keep = [owned.run / "client/state.json", owned.run / "tinker.db",
            owned.root / "ram/orbax/weights", owned.run / "checkpoints/partial.tmp",
            owned.run / "loras/adapter"]
    for path in keep:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep")
    assert clean(owned) == dict(files=1, bytes=owned.obj.size)
    assert not owned.path.exists()
    assert all(path.read_bytes() == b"keep" for path in keep)
    assert len(owned.calls) == 2
    assert "checkpoint_replica_reclaimed" in owned.log.read_text()
    assert clean(owned)["files"] == 0


@pytest.mark.parametrize("case", ["absent", "mismatch", "size", "generation", "disappeared", "error", "timeout"])
def test_remote_failure_or_change_preserves_local(owned, case):
    calls = 0
    def listing(*args, **kwargs):
        nonlocal calls
        calls += 1
        if case == "error":
            raise RuntimeError("GCS unavailable")
        if case == "timeout":
            import subprocess
            raise subprocess.TimeoutExpired("gcloud", 300)
        if case == "absent" or case == "disappeared" and calls == 2:
            return []
        if case == "mismatch":
            return [replace(owned.obj, md5="incorrect")]
        if case == "size":
            return [replace(owned.obj, size=999)]
        if case == "generation" and calls == 2:
            return [replace(owned.obj, generation="456")]
        return [owned.obj]
    owned.gcs.list = listing
    assert clean(owned)["files"] == 0
    assert owned.path.exists()


def test_composite_object_crc32c_is_verified(owned):
    crc = pytest.importorskip("google_crc32c")
    obj = replace(owned.obj, md5=None,
                  crc32c=base64.b64encode(crc.Checksum(owned.path.read_bytes()).digest()).decode())
    owned.gcs.list = lambda *args, **kwargs: [obj]
    assert clean(owned)["files"] == 1


def test_current_run_is_never_reclaimed(owned):
    assert retention.reclaim_checkpoints(owned.root, "old", owned.gcs, owned.log)["files"] == 0
    assert owned.path.exists() and not owned.calls


def test_held_lease_protects_active_run(owned):
    fd = retention.acquire_run_lease(owned.config, owned.root)
    try:
        assert clean(owned)["files"] == 0
        assert owned.path.exists() and not owned.calls
    finally:
        os.close(fd)


@pytest.mark.parametrize("after_hash", [False, True])
def test_orphan_or_newly_active_run_is_preserved(owned, monkeypatch, after_hash):
    checks = iter([False, True] if after_hash else [True])
    monkeypatch.setattr(retention, "run_is_active", lambda *args: next(checks))
    assert clean(owned)["files"] == 0
    assert owned.path.exists()


@pytest.mark.parametrize("kind", ["absent", "malformed", "list", "prefix", "root", "task", "run"])
def test_unknown_ownership_is_not_guessed(owned, kind):
    marker = owned.run / retention.MANIFEST
    if kind == "absent":
        marker.unlink()
    elif kind == "malformed":
        marker.write_text("{")
    elif kind == "list":
        marker.write_text("[]")
    else:
        record = json.loads(marker.read_text())
        key = {"prefix": "checkpoint_gcs", "root": "root", "task": "task_id", "run": "run_id"}[kind]
        record[key] = "unrelated"
        marker.write_text(json.dumps(record))
    assert clean(owned)["files"] == 0
    assert owned.path.exists() and not owned.calls


@pytest.mark.parametrize("kind", ["file", "directory", "run", "manifest", "hardlink"])
def test_linked_paths_are_not_deleted(owned, kind):
    if kind == "hardlink":
        os.link(owned.path, owned.root / "second-copy")
    else:
        path = {"file": owned.path, "directory": owned.path.parent,
                "run": owned.run, "manifest": owned.run / retention.MANIFEST}[kind]
        moved = owned.root / "unrelated"
        path.rename(moved)
        path.symlink_to(moved, target_is_directory=moved.is_dir())
    assert clean(owned)["files"] == 0
    assert owned.path.read_bytes() == b"checkpoint contents"


def test_changed_file_is_preserved(owned, monkeypatch):
    validate = retention.valid_file
    def mutate(path, obj):
        result = validate(path, obj)
        path.write_bytes(b"changed checkpoint")
        return result
    monkeypatch.setattr(retention, "valid_file", mutate)
    assert clean(owned)["files"] == 0
    assert owned.path.read_bytes() == b"changed checkpoint"


def test_cancel_or_disabled_cleanup_does_not_touch_files(owned):
    assert clean(owned, stopping=lambda: True)["files"] == 0
    assert clean(owned, timeout=0)["files"] == 0
    assert owned.path.exists() and not owned.calls


def test_reusing_run_id_cannot_change_durable_destination(owned):
    with pytest.raises(ValueError, match="ownership changed"):
        retention.acquire_run_lease(replace(owned.config, bucket="gs://another-bucket"), owned.root)


@pytest.mark.parametrize("value", [-1, "600", True])
def test_invalid_cleanup_timeout_is_rejected(value):
    config = Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32_budget.json")
    with pytest.raises(ValueError, match="checkpoint_cleanup_timeout"):
        replace(config, checkpoint_cleanup_timeout=value).validate()


def test_process_guard_detects_orphans_and_fails_closed(monkeypatch, tmp_path):
    psutil = pytest.importorskip("psutil")
    process = SimpleNamespace(uids=lambda: SimpleNamespace(real=os.getuid()),
                              status=lambda: "running", cmdline=lambda: [])
    monkeypatch.setattr(psutil, "process_iter", lambda: [process])
    record = dict(task_id="sky-managed-test_old_370-0", run_id="old")
    process.environ = lambda: {"SKYPILOT_TASK_ID": record["task_id"]}
    assert retention.run_is_active(record, tmp_path)
    process.environ = lambda: {"SKYPILOT_TASK_ID": "unrelated"}
    assert not retention.run_is_active(record, tmp_path)
    def denied():
        raise psutil.AccessDenied(123)
    process.environ = denied
    assert retention.run_is_active(record, tmp_path)


def test_preflight_reclaims_before_disk_check_without_touching_live_trainers(tmp_path, monkeypatch):
    from tpu.swarm.ray_train.host import Host
    import psutil
    import shutil
    config = Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32_budget.json")
    config = replace(config, retired_task_ids=[], retired_processes={})
    calls = []
    host = SimpleNamespace(config=config, root=tmp_path, gcs=None, rank=0,
                           log=tmp_path / "host.jsonl", stopping=SimpleNamespace(is_set=lambda: False),
                           clear_previous_adapter_exports=lambda: calls.append("adapters"),
                           heartbeat=lambda: {})
    monkeypatch.setattr(psutil, "process_iter", lambda *args: [])
    monkeypatch.setattr(retention, "reclaim_checkpoints", lambda *args, **kwargs: calls.append("checkpoints"))
    def disk(root):
        calls.append("disk")
        return SimpleNamespace(free=11 * 1024**3)
    monkeypatch.setattr(shutil, "disk_usage", disk)
    Host.preflight(host)
    assert calls == ["adapters", "checkpoints", "disk"]
    calls.clear()
    monkeypatch.setattr(psutil, "process_iter", lambda *args: [
        SimpleNamespace(pid=123, info={"cmdline": ["python", "-m", "skyrl.tinker.api"]})])
    with pytest.raises(RuntimeError, match="existing trainer"):
        Host.preflight(host)
    assert calls == []
