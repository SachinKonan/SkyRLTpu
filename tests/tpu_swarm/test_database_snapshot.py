import gzip
import json
import os
from pathlib import Path
import shutil
import sqlite3
import tarfile
import time

import pytest

from skyrl.tinker import db_models
from tpu.swarm.ray_train.database_snapshot import (
    TABLE, IncompleteDatabaseSnapshot, create_snapshot, restore_snapshot,
)


def queue(tmp_path, monkeypatch, status="PENDING"):
    blobs = tmp_path / "old-host-blobs"
    monkeypatch.setattr(db_models, "_FUTURE_BLOB_DIR", blobs)
    payload = {"data": [{"tokens": list(range(50000))}], "loss_fn": "importance_sampling"}
    codec = db_models.OffloadedJSON()
    ref = codec.process_bind_param(payload, None)
    assert "__blobref__" in ref
    db = tmp_path / "live.db"
    with sqlite3.connect(db) as conn:
        conn.execute("CREATE TABLE futures(request_id INTEGER PRIMARY KEY, status TEXT, request_data JSON, result_data JSON)")
        conn.execute("INSERT INTO futures VALUES (527, ?, ?, NULL)", (status, json.dumps(ref)))
    return db, blobs, payload, ref


def test_restore_on_new_host_preserves_large_training_payload(tmp_path, monkeypatch):
    live, old_blobs, payload, original_ref = queue(tmp_path, monkeypatch)
    backup = tmp_path / "backup.db"
    create_snapshot(live, backup)
    # Losing the original host must not lose the request.
    shutil.rmtree(old_blobs)
    live.unlink()
    restored = tmp_path / "new-host.db"
    restore_snapshot(backup, restored, tmp_path / "new-host-blobs")
    with sqlite3.connect(restored) as conn:
        ref = json.loads(conn.execute("SELECT request_data FROM futures").fetchone()[0])
        assert not conn.execute("SELECT 1 FROM sqlite_master WHERE name=?", (TABLE,)).fetchone()
    assert ref != original_ref
    assert db_models.OffloadedJSON().process_result_value(ref, None) == payload


def test_payload_older_than_two_hours_survives_new_writes(tmp_path, monkeypatch):
    _, _, payload, ref = queue(tmp_path, monkeypatch)
    old = time.time() - 24 * 3600
    os.utime(ref["__blobref__"], (old, old))
    codec = db_models.OffloadedJSON()
    codec.process_bind_param(payload, None)
    assert codec.process_result_value(ref, None) == payload


def test_missing_payload_raises_instead_of_returning_none(tmp_path, monkeypatch):
    _, blobs, _, ref = queue(tmp_path, monkeypatch)
    shutil.rmtree(blobs)
    with pytest.raises(db_models.MissingFuturePayloadError, match="restore a complete"):
        db_models.OffloadedJSON().process_result_value(ref, None)


def test_incomplete_backup_keeps_previous_snapshot(tmp_path, monkeypatch):
    live, blobs, _, _ = queue(tmp_path, monkeypatch)
    backup = tmp_path / "backup.db"
    create_snapshot(live, backup)
    previous = backup.read_bytes()
    shutil.rmtree(blobs)
    with pytest.raises(IncompleteDatabaseSnapshot, match="request 527"):
        create_snapshot(live, backup)
    assert backup.read_bytes() == previous


def test_legacy_missing_pending_payload_blocks_restore_without_installing_db(tmp_path, monkeypatch):
    live, blobs, _, _ = queue(tmp_path, monkeypatch)
    shutil.rmtree(blobs)
    restored = tmp_path / "restored.db"
    with pytest.raises(IncompleteDatabaseSnapshot, match="legacy backup"):
        restore_snapshot(live, restored, tmp_path / "blobs")
    assert not restored.exists()


def test_legacy_terminal_request_can_be_retired_without_losing_result(tmp_path, monkeypatch):
    live, blobs, _, _ = queue(tmp_path, monkeypatch, status="COMPLETED")
    with sqlite3.connect(live) as db:
        db.execute("UPDATE futures SET result_data=?", (json.dumps({"loss": 1.25}),))
    shutil.rmtree(blobs)
    backup, restored = tmp_path / "backup.db", tmp_path / "restored.db"
    create_snapshot(live, backup)
    restore_snapshot(backup, restored, tmp_path / "blobs")
    with sqlite3.connect(restored) as db:
        status, request, result = db.execute("SELECT status, request_data, result_data FROM futures").fetchone()
    assert status == "COMPLETED"
    assert json.loads(request) is None
    assert json.loads(result) == {"loss": 1.25}


@pytest.mark.parametrize("damage", ["missing", "corrupt"])
def test_incomplete_embedded_snapshot_never_installs_db(tmp_path, monkeypatch, damage):
    live, _, _, _ = queue(tmp_path, monkeypatch)
    backup = tmp_path / "backup.db"
    create_snapshot(live, backup)
    with sqlite3.connect(backup) as db:
        db.execute(f"DELETE FROM {TABLE}" if damage == "missing" else f"UPDATE {TABLE} SET data=x'1234'")
    restored = tmp_path / "restored.db"
    with pytest.raises(IncompleteDatabaseSnapshot):
        restore_snapshot(backup, restored, tmp_path / "blobs")
    assert not restored.exists()


def test_default_v5p_bundle_includes_storage_fix(tmp_path):
    from tpu.swarm.ray_train.build import build
    from tpu.swarm.ray_train.commands import trainer_environment
    from tpu.swarm.ray_train.config import Config
    from tpu.swarm.ray_train.overlay import install
    profile = Path("tpu/swarm/ray_train/profiles/qwen_v5p_32.json")
    cfg = Config.load(profile)
    archive, _, _ = build(profile, tmp_path / "build")
    with tarfile.open(archive) as bundle:
        prefix = "tpu/swarm/ray_train/source_overlay/"
        manifest = json.load(bundle.extractfile(prefix + "manifest.json"))
        assert "skyrl/tinker/db_models.py" in manifest
        overlay = tmp_path / "overlay"
        overlay.mkdir()
        (overlay / "manifest.json").write_text(json.dumps(manifest))
        for name in manifest:
            target = overlay / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(bundle.extractfile(prefix + name).read())
        assert "tpu/swarm/ray_train/database_snapshot.py" in bundle.getnames()
    install(overlay, tmp_path / "source")
    env = trainer_environment(cfg, tmp_path, tmp_path / "run", ["10.0.0.1"], 0)
    assert env["SKYRL_FUTURE_BLOB_DIR"] == str(tmp_path / "run/future-blobs")
