"""Portable SQLite backups including compressed, locally offloaded payloads.

The auxiliary table exists only in the backup. It makes one GCS object the
publication boundary for both the queue and its payloads, without expanding
large JSON into the live database or relying on files from a previous VM.
"""
from __future__ import annotations

import gzip
from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3


TABLE = "skyrl_snapshot_payloads"


def require_checkpoint_client(client: Path, run_id: str, member: str, minimum: int):
    """Refuse a fresh client when this launch explicitly requires saved work."""
    logs = client / 'tinker_log' / run_id
    rows = [json.loads(line) for line in
            (logs / f'member_{member}' / 'checkpoints.jsonl').read_text().splitlines() if line.strip()]
    latest = rows[-1] if rows else {}
    step = latest.get('batch', -1)
    if step < minimum or not latest.get('state_path'):
        raise RuntimeError(f'required checkpoint >= {minimum}, found {step}')
    pool = logs / f'puct_sampler_step_{step:06d}.json'
    if not pool.is_file():
        raise RuntimeError(f'checkpoint {step} has no matching search snapshot')
    return step


def abandon_pending_for_checkpoint_resume(path: Path) -> int:
    """Retire dead-client requests before a checkpoint-based client restart.

    Completed requests and model/checkpoint registrations remain intact.
    Call only before starting the API; never against a running trainer.
    """
    with closing(sqlite3.connect(path)) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='futures'").fetchone():
            return 0
        result = json.dumps({'error': 'Interrupted request superseded by checkpoint resume'})
        cursor = db.execute(
            "UPDATE futures SET status='FAILED', result_data=?, completed_at=CURRENT_TIMESTAMP "
            "WHERE status='PENDING'", (result,))
        count = cursor.rowcount
        db.commit()
        return count


class IncompleteDatabaseSnapshot(RuntimeError):
    pass


def _references(db):
    if not db.execute("SELECT 1 FROM sqlite_master WHERE name='futures'").fetchone():
        return
    for rid, status, request, result in db.execute(
        "SELECT request_id, status, request_data, result_data FROM futures"
    ):
        for column, raw in (("request_data", request), ("result_data", result)):
            value = json.loads(raw) if isinstance(raw, str) else raw
            if str(status).lower() == "pending" and column == "request_data" and value is None:
                raise IncompleteDatabaseSnapshot(f"Pending request {rid} has no request_data")
            if isinstance(value, dict) and "__blobref__" in value:
                yield rid, str(status).lower(), column, value["__blobref__"]


def _validate(data, rid, path):
    try:
        json.loads(gzip.decompress(data))
    except (OSError, ValueError, EOFError) as exc:
        raise IncompleteDatabaseSnapshot(f"Corrupt payload for request {rid}: {path}") from exc


def create_snapshot(source: Path, destination: Path):
    """Publish a complete local snapshot; leave the previous one on failure."""
    source, destination = Path(source).resolve(), Path(destination).resolve()
    staging = destination.with_suffix(".partial")
    staging.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as live:
            with closing(sqlite3.connect(staging)) as backup:
                live.backup(backup)
                backup.execute("PRAGMA journal_mode=DELETE")
                backup.execute(f"DROP TABLE IF EXISTS {TABLE}")
                backup.execute(f"CREATE TABLE {TABLE}(path TEXT PRIMARY KEY, data BLOB NOT NULL)")
                for rid, status, column, path in list(_references(backup)):
                    if backup.execute(f"SELECT 1 FROM {TABLE} WHERE path=?", (path,)).fetchone():
                        continue
                    try:
                        data = Path(path).read_bytes()
                    except OSError as exc:
                        # Old deployments already GC'd terminal request bodies.
                        # They are not replayed; pending data and completed
                        # results must never be silently discarded.
                        if status in ("completed", "failed") and column == "request_data":
                            backup.execute("UPDATE futures SET request_data='null' WHERE request_id=?", (rid,))
                            continue
                        raise IncompleteDatabaseSnapshot(
                            f"Missing {column} for {status} request {rid}: {path}; "
                            "resume from a consistent training checkpoint, not this incomplete queue"
                        ) from exc
                    _validate(data, rid, path)
                    backup.execute(f"INSERT INTO {TABLE} VALUES (?, ?)", (path, data))
                backup.commit()
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)


def restore_snapshot(source: Path, destination: Path, blob_dir: Path):
    """Validate/materialize payloads before installing the database for startup.

    Legacy backups are accepted only when their required local blobs still
    exist. A failed restore never leaves a database that startup can replay.
    """
    source, destination, blob_dir = Path(source).resolve(), Path(destination).resolve(), Path(blob_dir).resolve()
    staging = destination.with_suffix(".restore-partial")
    staging.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(source.as_uri() + "?mode=ro", uri=True)) as original:
            with closing(sqlite3.connect(staging)) as db:
                original.backup(db)
                db.execute("PRAGMA journal_mode=DELETE")
                embedded = db.execute("SELECT 1 FROM sqlite_master WHERE name=?", (TABLE,)).fetchone()
                for rid, status, column, old_path in list(_references(db)):
                    row = db.execute(f"SELECT data FROM {TABLE} WHERE path=?", (old_path,)).fetchone() if embedded else None
                    if embedded and row is None:
                        raise IncompleteDatabaseSnapshot(f"Snapshot is missing embedded payload for request {rid}: {old_path}")
                    try:
                        data = row[0] if row else Path(old_path).read_bytes()
                    except OSError as exc:
                        if status in ("completed", "failed") and column == "request_data":
                            db.execute("UPDATE futures SET request_data='null' WHERE request_id=?", (rid,))
                            continue
                        raise IncompleteDatabaseSnapshot(
                            f"Cannot restore {status} request {rid}: missing {column} {old_path}. "
                            "The legacy backup did not preserve payloads; resume from a consistent training checkpoint."
                        ) from exc
                    _validate(data, rid, old_path)
                    blob_dir.mkdir(parents=True, exist_ok=True)
                    path = blob_dir / (hashlib.sha256(data).hexdigest() + ".json.gz")
                    partial = path.with_suffix(".partial")
                    partial.write_bytes(data)
                    partial.replace(path)
                    db.execute(f"UPDATE futures SET {column}=? WHERE request_id=?",
                               (json.dumps({"__blobref__": str(path)}), rid))
                if embedded:
                    db.execute(f"DROP TABLE {TABLE}")
                db.commit()
                # Dropping the embedded table leaves its pages allocated.
                db.execute("VACUUM")
        staging.replace(destination)
    finally:
        staging.unlink(missing_ok=True)
