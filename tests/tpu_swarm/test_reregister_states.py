"""Checkpoint registration tolerates the API's short-lived SQLite writers."""

import sqlite3
import subprocess
import sys
import time
from pathlib import Path


def test_checkpoint_registration_waits_for_startup_writer(tmp_path):
    path = tmp_path / "registry.db"
    with sqlite3.connect(path) as db:
        db.executescript('''
            CREATE TABLE sessions (session_id TEXT PRIMARY KEY, tags TEXT, user_metadata TEXT,
              sdk_version TEXT, status TEXT, created_at TEXT, heartbeat_count INTEGER);
            CREATE TABLE models (model_id TEXT PRIMARY KEY, base_model TEXT, lora_config TEXT,
              status TEXT, request_id INTEGER, session_id TEXT, created_at TEXT);
            CREATE TABLE checkpoints (model_id TEXT, checkpoint_id TEXT, checkpoint_type TEXT,
              status TEXT, created_at TEXT, completed_at TEXT,
              PRIMARY KEY(model_id, checkpoint_id, checkpoint_type));
        ''')
    archive = tmp_path / "checkpoints/model_abc/000003.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"checkpoint presence fixture")
    script = Path(__file__).parents[2] / "tpu/reregister_states.py"
    writer = sqlite3.connect(path)
    writer.execute("BEGIN IMMEDIATE")
    child = subprocess.Popen(
        [sys.executable, str(script), "--db", str(path),
         "--ckpt-root", str(tmp_path / "checkpoints"),
         "--base-model", "test-model", "--entry", "model_abc:000003"],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        # Reproduce contention longer than sqlite3's previous five-second limit.
        time.sleep(7)
        writer.commit()
        stdout, stderr = child.communicate(timeout=10)
        assert child.returncode == 0, stdout + stderr
        assert writer.execute("SELECT model_id FROM checkpoints").fetchall() == [("model_abc",)]
    finally:
        writer.close()
        if child.poll() is None:
            child.kill()
            child.wait()
