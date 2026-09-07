"""Reclaim verified checkpoint replicas from inactive, executor-owned runs."""
from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import time

from .cache import valid_file
from .events import emit

MANIFEST = ".checkpoint-owner.json"
LEASE = ".checkpoint-owner.lock"


def private_path(root, path):
    relative = path.relative_to(root)
    current = root
    for part in relative.parts:
        current /= part
        if current.is_symlink():
            raise ValueError("checkpoint retention refuses symlinks")
    if not path.resolve().is_relative_to(root):
        raise ValueError("checkpoint path escapes executor root")
    return path


def acquire_run_lease(config, root):
    """Bootstrap holds this lease until services and final writebacks stop."""
    run = private_path(root, root / "runs" / config.run_id)
    run.mkdir(parents=True, exist_ok=True)
    fd = os.open(run / LEASE, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = private_path(root, run / MANIFEST)
        record = dict(version=1, root=str(root), run_id=config.run_id,
                      checkpoint_gcs=config.run_gcs + "/checkpoints",
                      task_id=os.environ.get("SKYPILOT_TASK_ID", ""))
        if manifest.exists():
            previous = json.loads(manifest.read_text())
            if not isinstance(previous, dict) or any(previous.get(key) != record[key] for key in
                   ("version", "root", "run_id", "checkpoint_gcs")):
                raise ValueError("checkpoint ownership changed for existing run")
        stage = private_path(root, run / (MANIFEST + ".partial"))
        stage.write_text(json.dumps(record))
        stage.replace(manifest)
        return fd
    except BaseException:
        os.close(fd)
        raise


def run_is_active(record, run):
    """An orphan can outlive bootstrap's lock. Unknown process access is unsafe."""
    import psutil
    for process in psutil.process_iter():
        try:
            if process.uids().real != os.getuid() or process.status() == psutil.STATUS_ZOMBIE:
                continue
            env = process.environ()
            if (env.get("SKYPILOT_TASK_ID") == record["task_id"]
                    or env.get("RAY_NAMESPACE") == record["run_id"]
                    or env.get("TTD_RUN_DIR") == str(run / "client")
                    or any(str(run) in arg for arg in process.cmdline())):
                return True
        except psutil.NoSuchProcess:
            continue
        except psutil.AccessDenied:
            return True
    return False


def file_identity(root, path):
    info = private_path(root, path).stat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("checkpoint is not a private regular file")
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def reclaim_checkpoints(root, current_run, gcs, log, timeout=600, stopping=lambda: False):
    """Never delete GCS objects, active state, unknown runs or unverified bytes."""
    root = Path(root).resolve()
    runs = private_path(root, root / "runs")
    deadline = time.monotonic() + timeout
    removed = total = 0

    def check_deadline():
        if stopping() or time.monotonic() >= deadline:
            raise TimeoutError("checkpoint cleanup stopped or exceeded its time budget")

    if timeout <= 0 or not runs.exists():
        return dict(files=0, bytes=0)
    for run in sorted(runs.iterdir()):
        if run.name == current_run:
            continue
        fd = None
        try:
            check_deadline()
            private_path(root, run)
            if not run.is_dir():
                continue
            manifest = private_path(root, run / MANIFEST)
            if not manifest.is_file():
                emit(log, "checkpoint_cleanup_skipped", run_id=run.name, reason="no ownership record")
                continue
            # A missing lease indicates an older/unknown ownership protocol.
            fd = os.open(run / LEASE, os.O_RDWR | os.O_NOFOLLOW)
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            record = json.loads(manifest.read_text())
            if not isinstance(record, dict):
                raise ValueError("invalid checkpoint ownership record")
            prefix = record.get("checkpoint_gcs", "")
            if (record.get("version") != 1 or record.get("root") != str(root)
                    or record.get("run_id") != run.name
                    or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,100}", run.name)
                    or not re.fullmatch(r"sky-managed-[^\s]+_\d+-\d+", record.get("task_id", ""))
                    or not re.fullmatch(r"gs://[a-z0-9._-]+/ray-training/" + re.escape(run.name) + r"/checkpoints", prefix)):
                raise ValueError("invalid checkpoint ownership record")
            if run_is_active(record, run):
                raise RuntimeError("run still has live or uninspectable processes")
            directory = private_path(root, run / "checkpoints")
            if not directory.exists():
                continue
            objects = {obj.relative: obj for obj in gcs.list(prefix, allow_empty=True)}
            verified = []
            for path in sorted(directory.rglob("*.tar.gz")):
                check_deadline()
                before = file_identity(root, path)
                obj = objects.get(path.relative_to(directory).as_posix())
                if obj is None or not valid_file(path, obj):
                    emit(log, "checkpoint_cleanup_preserved", local_path=str(path), reason="missing or mismatched GCS copy")
                    continue
                if file_identity(root, path) != before:
                    raise RuntimeError("checkpoint changed during verification")
                verified.append((path, obj, before))
            if not verified:
                continue
            check_deadline()
            # Revalidate generations after hashing, before deleting any local copy.
            latest = {obj.relative: obj for obj in gcs.list(prefix, allow_empty=True)}
            if any(latest.get(obj.relative) != obj for _, obj, _ in verified):
                raise RuntimeError("GCS checkpoint changed during verification")
            if run_is_active(record, run):
                raise RuntimeError("run became active during verification")
            for path, obj, before in verified:
                check_deadline()
                if file_identity(root, path) != before:
                    raise RuntimeError("checkpoint changed before removal")
                path.unlink()
                removed += 1
                total += obj.size
                emit(log, "checkpoint_replica_reclaimed", local_path=str(path), gcs=obj.uri,
                     generation=obj.generation, bytes=obj.size)
        except TimeoutError as exc:
            emit(log, "checkpoint_cleanup_skipped", run_id=run.name, reason=str(exc))
            break
        except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
            emit(log, "checkpoint_cleanup_skipped", run_id=run.name, reason=str(exc))
        finally:
            if fd is not None:
                os.close(fd)
    result = dict(files=removed, bytes=total)
    emit(log, "checkpoint_cleanup_complete", **result)
    return result
