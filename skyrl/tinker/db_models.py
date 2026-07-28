"""Database models for the Tinker API."""

import gzip
import json
import os
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path

from sqlalchemy import DateTime, event
from sqlalchemy.engine import url as sqlalchemy_url
from sqlalchemy.types import TypeDecorator
from sqlmodel import JSON, Field, SQLModel

from skyrl.tinker import types


def enable_sqlite_wal(engine) -> None:
    """Enable WAL mode and busy timeout for SQLite engines.

    WAL mode allows concurrent readers with a single writer.
    Busy timeout makes SQLite retry internally instead of immediately
    raising 'database is locked'.

    No-op for non-SQLite engines.
    """
    if engine.dialect.name != "sqlite":
        return

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA busy_timeout=30000")
        # Cap the WAL file: SQLite truncates it back to this size on each
        # checkpoint instead of letting it grow unboundedly (which made every
        # DB op crawl). With OffloadedJSON keeping rows tiny the WAL rarely
        # approaches this, but it's a hard safety bound.
        cursor.execute("PRAGMA journal_size_limit=67108864")  # 64 MB
        cursor.close()


# --- Large-payload offload --------------------------------------------------
# forward_backward requests carry per-token tensors for the whole batch; at long
# (thinking-mode) sequence lengths that is ~140MB of JSON PER STEP. Storing that
# in a SQLite TEXT column bloats the WAL and stalls the future queue (the
# engine/API deadlock we hit — small mathrl requests never triggered it). We
# spill large payloads to gzipped files on LOCAL disk and keep only a tiny
# {"__blobref__": path} reference in the row, so the DB stays small and the queue
# scales to any sequence length. Same-host only (API writes, engine reads the
# same local file). Transparent to callers — they still read/write plain dicts.
_FUTURE_BLOB_DIR = Path(os.environ.get("SKYRL_FUTURE_BLOB_DIR", "/tmp/skyrl-future-blobs"))
_FUTURE_BLOB_THRESHOLD = 256 * 1024  # spill payloads whose JSON exceeds this many bytes
_FUTURE_BLOB_TTL_SEC = 2 * 3600      # opportunistically GC blob files older than this


def _gc_future_blobs() -> None:
    """Best-effort delete of stale blob files. A request/result blob is consumed
    within a step (~minutes), so anything older than the TTL is safe to remove."""
    try:
        cutoff = time.time() - _FUTURE_BLOB_TTL_SEC
        for p in _FUTURE_BLOB_DIR.glob("*.json.gz"):
            try:
                if p.stat().st_mtime < cutoff:
                    p.unlink()
            except OSError:
                pass
    except OSError:
        pass


class OffloadedJSON(TypeDecorator):
    """JSON column that spills large payloads to gzipped files on local disk,
    storing only a small reference in the row (see the note above)."""

    impl = JSON
    cache_ok = True

    def process_bind_param(self, value, dialect):
        if value is None:
            return None
        try:
            payload = json.dumps(value)
        except (TypeError, ValueError):
            return value  # let the JSON impl handle/raise as before
        if len(payload) <= _FUTURE_BLOB_THRESHOLD:
            return value  # small: store inline as ordinary JSON
        _FUTURE_BLOB_DIR.mkdir(parents=True, exist_ok=True)
        path = _FUTURE_BLOB_DIR / f"{uuid.uuid4().hex}.json.gz"
        with gzip.open(path, "wt", encoding="utf-8", compresslevel=1) as f:
            f.write(payload)
        _gc_future_blobs()
        return {"__blobref__": str(path)}

    def process_result_value(self, value, dialect):
        if isinstance(value, dict) and "__blobref__" in value:
            try:
                with gzip.open(value["__blobref__"], "rt", encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                return None  # blob missing/corrupt (GC'd stale future) -> treat as gone
        return value


def get_async_database_url(db_url: str) -> str:
    """Get the async database URL.

    Args:
        db_url: Optional database URL to use.

    Returns:
        Async database URL string for SQLAlchemy.

    Raises:
        ValueError: If the database scheme is not supported.
    """
    parsed_url = sqlalchemy_url.make_url(db_url)

    match parsed_url.get_backend_name():
        case "sqlite":
            async_url = parsed_url.set(drivername="sqlite+aiosqlite")
        case "postgresql":
            async_url = parsed_url.set(drivername="postgresql+asyncpg")
        case _ if "+" in parsed_url.drivername:
            # Already has an async driver specified, keep it
            async_url = parsed_url
        case backend_name:
            raise ValueError(f"Unsupported database scheme: {backend_name}")

    return async_url.render_as_string(hide_password=False)


class RequestStatus(str, Enum):
    """Status of a request."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


class CheckpointStatus(str, Enum):
    """Status of a checkpoint."""

    PENDING = "pending"
    COMPLETED = "completed"
    FAILED = "failed"


# SQLModel table definitions
class ModelDB(SQLModel, table=True):
    __tablename__ = "models"

    model_id: str = Field(primary_key=True)
    base_model: str
    lora_config: dict[str, object] = Field(sa_type=JSON)
    status: str = Field(index=True)
    request_id: int
    session_id: str = Field(foreign_key="sessions.session_id", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))


class FutureDB(SQLModel, table=True):
    __tablename__ = "futures"

    request_id: int | None = Field(default=None, primary_key=True, sa_column_kwargs={"autoincrement": True})
    request_type: types.RequestType
    model_id: str | None = Field(default=None, index=True)
    request_data: dict = Field(sa_type=OffloadedJSON)  # types.{request_type}Input; large payloads spill to disk
    result_data: dict | None = Field(default=None, sa_type=OffloadedJSON)  # types.{request_type}Output
    status: RequestStatus = Field(default=RequestStatus.PENDING, index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))
    completed_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))


class CheckpointDB(SQLModel, table=True):
    __tablename__ = "checkpoints"

    model_id: str = Field(foreign_key="models.model_id", primary_key=True)
    checkpoint_id: str = Field(primary_key=True)
    checkpoint_type: types.CheckpointType = Field(primary_key=True)
    status: CheckpointStatus
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))
    completed_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))
    error_message: str | None = None


class SessionDB(SQLModel, table=True):
    __tablename__ = "sessions"

    session_id: str = Field(primary_key=True)
    tags: list[str] = Field(default_factory=list, sa_type=JSON)
    user_metadata: dict = Field(default_factory=dict, sa_type=JSON)
    sdk_version: str
    status: str = Field(default="active", index=True)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))
    last_heartbeat_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True), index=True)
    heartbeat_count: int = 0


class SamplingSessionDB(SQLModel, table=True):
    __tablename__ = "sampling_sessions"

    sampling_session_id: str = Field(primary_key=True)
    session_id: str = Field(foreign_key="sessions.session_id", index=True)
    sampling_session_seq_id: int
    base_model: str | None = None
    model_path: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))


class EngineStateDB(SQLModel, table=True):
    """Engine→API handoff for the inference engine the backend stands up.

    Singleton row (``singleton_id=1``). Written by the backend when a new
    inference client is built (or torn down) and read by the API's
    forwarding client to resolve the vLLM proxy URL.
    """

    __tablename__ = "engine_state"

    singleton_id: int = Field(default=1, primary_key=True)

    # Proxy URL of the engine-managed vLLM. None when no vLLM has been
    # stood up yet (no create_model, FFT path, or last delete tore down).
    inference_proxy_url: str | None = None

    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc), sa_type=DateTime(timezone=True))
