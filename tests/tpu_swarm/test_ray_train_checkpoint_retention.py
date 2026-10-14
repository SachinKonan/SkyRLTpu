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


def test_preflight_reclaims_storage_before_disk_check(tmp_path, monkeypatch):
    from tpu.swarm.ray_train.host import Host
    import psutil
    import shutil
    config = Config.load("tpu/swarm/ray_train/profiles/qwen_v5p_32_budget.json")
    config = replace(config, retired_task_ids=[], retired_processes={})
    calls = []
    host = SimpleNamespace(config=config, root=tmp_path, gcs=None, rank=0,
                           log=tmp_path / "host.jsonl", stopping=SimpleNamespace(is_set=lambda: False),
                           clear_previous_adapter_exports=lambda: calls.append("adapters"),
                           reconcile_storage=lambda startup: calls.append(("storage", startup)),
                           heartbeat=lambda: {})
    def disk(root):
        calls.append("disk")
        return SimpleNamespace(free=11 * 1024**3)
    monkeypatch.setattr(shutil, "disk_usage", disk)
    Host.preflight(host)
    # Storage (verified checkpoints, then other tasks' leftovers) is reclaimed before the free-disk check.
    # Leftover trainers/engines no longer reach preflight: lifecycle.prepare stops them (or refuses on a
    # shared host) before the runtime unit starts; see test_lifecycle_wiring.py.
    assert calls == ["adapters", ("storage", True), "disk"]


@pytest.fixture
def current(tmp_path):
    run = tmp_path / 'runs/current'
    ledger = run / 'client/tinker_log/current/member_qwen/checkpoints.jsonl'
    ledger.parent.mkdir(parents=True)
    records, objects, paths = [], [], []
    for step in range(1, 5):
        name = f'{step:06d}'
        records.append(dict(batch=step, state_path=f'tinker://model_a/weights/{name}',
                            sampler_path=f'tinker://model_a/{name}'))
        for family in ['', 'sampler_weights/']:
            relative = f'model_a/{family}{name}.tar.gz'
            path = run / 'checkpoints' / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(relative.encode())
            paths.append(path)
            objects.append(Object('gs://bucket/run/checkpoints/' + relative, relative,
                                  path.stat().st_size, '1',
                                  base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode(), None))
    # Step 4 is in flight and has not been acknowledged by the client.
    ledger.write_text('\n'.join(json.dumps(r) for r in records[:3]))
    return SimpleNamespace(run=run, ledger=ledger, objects=objects, paths=paths,
                           gcs=SimpleNamespace(list=lambda *a, **kw: objects), log=run/'cleanup.jsonl')


def test_current_retention_keeps_two_batches_and_unacknowledged_files(current):
    result = retention.reclaim_completed_checkpoints(current.run, 'gs://bucket/run', current.gcs, current.log)
    assert result['files'] == 2
    assert all(not p.exists() for p in current.paths[:2])
    assert all(p.exists() for p in current.paths[2:])
    assert current.ledger.exists()


@pytest.mark.parametrize('failure', ['checksum', 'generation', 'malformed', 'symlink'])
def test_current_retention_fails_closed(current, failure, tmp_path):
    if failure == 'checksum':
        current.objects[:2] = [replace(o, md5='wrong') for o in current.objects[:2]]
    elif failure == 'generation':
        calls = []
        def listing(*args, **kwargs):
            calls.append(1)
            return current.objects if len(calls) == 1 else [replace(o, generation='2') for o in current.objects]
        current.gcs.list = listing
    elif failure == 'malformed':
        with current.ledger.open('a') as stream:
            stream.write('\n{"batch":')
    else:
        outside = tmp_path/'protected';outside.write_bytes(current.paths[0].read_bytes())
        current.paths[0].unlink();current.paths[0].symlink_to(outside)
    result = retention.reclaim_completed_checkpoints(current.run, 'gs://bucket/run', current.gcs, current.log)
    assert result['files'] == 0
    assert all(p.exists() for p in current.paths)


def test_other_rank_applies_rank_zero_plan_without_a_ledger(current, tmp_path):
    plan = retention.checkpoint_plan(current.run)
    # The assigned trainer leader holds the archives but has no client ledger.
    other = tmp_path / 'leader/runs/current'
    for path in current.paths:
        copy = other / path.relative_to(current.run)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    result = retention.reclaim_completed_checkpoints(other, 'gs://bucket/run', current.gcs, current.log,
                                                     plan=json.loads(json.dumps(plan)))
    assert result['files'] == 2
    left = sorted(p.relative_to(other).as_posix() for p in other.rglob('*.tar.gz'))
    assert left == sorted(p.relative_to(current.run).as_posix() for p in current.paths[2:])


@pytest.mark.parametrize('plan', [None, {'version': 2, 'eligible': [], 'protected': []},
                                  {'version': 1, 'eligible': ['../escape.tar.gz'], 'protected': []},
                                  {'version': 1, 'eligible': ['model_a/000001.tar.gz']}])
def test_malformed_plan_deletes_nothing(current, tmp_path, plan):
    other = tmp_path / 'leader/runs/current'
    for path in current.paths:
        copy = other / path.relative_to(current.run)
        copy.parent.mkdir(parents=True, exist_ok=True)
        copy.write_bytes(path.read_bytes())
    plan = plan if plan is not None else 'not a plan'
    result = retention.reclaim_completed_checkpoints(other, 'gs://bucket/run', current.gcs, current.log, plan=plan)
    assert result['files'] == 0 and len(list(other.rglob('*.tar.gz'))) == len(current.paths)


def test_completed_cleanup_stops_at_its_deadline(current, monkeypatch):
    hashed = []
    def slow_valid(path, obj):
        hashed.append(path)
        return True
    monkeypatch.setattr(retention, 'valid_file', slow_valid)
    result = retention.reclaim_completed_checkpoints(current.run, 'gs://bucket/run', current.gcs, current.log,
                                                     stopping=lambda: len(hashed) >= 1)
    assert result['files'] == 0 and all(p.exists() for p in current.paths)
    assert 'time budget' in current.log.read_text()
