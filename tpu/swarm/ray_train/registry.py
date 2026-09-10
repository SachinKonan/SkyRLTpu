"""Durable Tinker registry helpers for the Ray v2 executor.

The API server keeps its models/checkpoints registry in a per-run sqlite
file on the trainer host. Checkpoint tarballs are mirrored to GCS, but a
reclaimed VM loses the registry, and a fresh server then answers
`Model not found` for weights that exist (job 596). Two things keep resume
working: back the database up with its WAL folded in (job 578's backups
uploaded an empty main file while every row sat in `tinker-backup.db-wal`),
and re-register whatever the client's checkpoint log says is durable.
"""
from __future__ import annotations

import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

SESSION_ID = "session_reregister"
_STATE = re.compile(r"^tinker://(model_[0-9a-f]+)/weights/([^/]+)$")


def _now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S.%f")


def backup_database(source: Path, destination: Path):
    """Copy a live sqlite database into a self-contained file.

    `Connection.backup` copies page 1, so the destination inherits WAL mode
    and the copied rows land in `<destination>-wal` until a checkpoint;
    switching the journal mode back folds them into the main file, which is
    the only file the writeback uploads.
    """
    src = sqlite3.connect(source)
    dst = sqlite3.connect(destination)
    try:
        src.backup(dst)
        dst.execute("PRAGMA journal_mode=DELETE")
        dst.commit()
    finally:
        dst.close()
        src.close()


def harvest_entries(paths) -> list[tuple[str, str]]:
    """(model_id, checkpoint_id) pairs from the client's checkpoints.jsonl files."""
    found = set()
    for path in paths:
        for line in Path(path).read_text().splitlines():
            try:
                row = json.loads(line)
            except ValueError:
                continue
            match = _STATE.match(row.get("state_path") or "")
            if match:
                found.add((match.group(1), match.group(2)))
    return sorted(found)


def register_rows(db_path: Path, entries, base_model: str, lora_config: dict, present) -> list[str]:
    """Insert models/checkpoints rows for entries whose tarballs `present`
    confirms (called with model_id, checkpoint_id, subdir in {"", "sampler_weights"}).
    Returns the registered "model/ckpt TYPE" labels; existing rows are kept."""
    registered = []
    db = sqlite3.connect(db_path, timeout=30)
    try:
        db.execute("INSERT OR IGNORE INTO sessions (session_id, tags, user_metadata, sdk_version, status,"
                   " created_at, heartbeat_count) VALUES (?, '[]', '{}', 'reregister', 'active', ?, 0)",
                   (SESSION_ID, _now()))
        for model_id, ckpt in sorted(set(entries)):
            if not present(model_id, ckpt, ""):
                continue
            db.execute("INSERT OR IGNORE INTO models (model_id, base_model, lora_config, status, request_id,"
                       " session_id, created_at) VALUES (?, ?, ?, 'created', 0, ?, ?)",
                       (model_id, base_model, json.dumps(lora_config), SESSION_ID, _now()))
            db.execute("INSERT OR IGNORE INTO checkpoints (model_id, checkpoint_id, checkpoint_type, status,"
                       " created_at, completed_at) VALUES (?, ?, 'TRAINING', 'COMPLETED', ?, ?)",
                       (model_id, ckpt, _now(), _now()))
            registered.append(f"{model_id}/{ckpt} TRAINING")
            if present(model_id, ckpt, "sampler_weights"):
                db.execute("INSERT OR IGNORE INTO checkpoints (model_id, checkpoint_id, checkpoint_type, status,"
                           " created_at, completed_at) VALUES (?, ?, 'SAMPLER', 'COMPLETED', ?, ?)",
                           (model_id, ckpt, _now(), _now()))
                registered.append(f"{model_id}/{ckpt} SAMPLER")
        db.commit()
    finally:
        db.close()
    return registered
