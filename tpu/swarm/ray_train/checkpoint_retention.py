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


PLAN = "checkpoint-plan.json"


def checkpoint_plan(run, keep=2):
    """Archive paths named by the client ledger: eligible ones and the newest-two protected ones.

    Only rank 0 has the ledger (the client runs there); it publishes this plan so other ranks holding
    archives (an assigned trainer leader) apply the same rules without reading the ledger.
    """
    run = Path(run).resolve()
    if keep < 2:
        raise ValueError("retain at least two acknowledged checkpoint batches")
    eligible, protected = set(), set()
    for ledger in (run / "client" / "tinker_log").glob("*/member_*/checkpoints.jsonl"):
        private_path(run, ledger)
        records = [json.loads(line) for line in ledger.read_text().splitlines() if line.strip()]
        if any(type(r.get("batch")) is not int or r["batch"] < 0 for r in records):
            raise ValueError("invalid checkpoint ledger")
        recent = set(sorted({r["batch"] for r in records})[-keep:])
        for record in records:
            for key, pattern, family in (
                ("state_path", r"tinker://([A-Za-z0-9_-]+)/weights/([A-Za-z0-9_-]+)", ""),
                ("sampler_path", r"tinker://([A-Za-z0-9_-]+)/([A-Za-z0-9_-]+)", "sampler_weights/"),
            ):
                match = re.fullmatch(pattern, record.get(key, ""))
                if not match:
                    raise ValueError("invalid checkpoint ledger path")
                relative = f"{match[1]}/{family}{match[2]}.tar.gz"
                (protected if record["batch"] in recent else eligible).add(relative)
    return dict(version=1, eligible=sorted(eligible), protected=sorted(protected))


def valid_plan(plan):
    path = r"[A-Za-z0-9_-]+/(?:sampler_weights/)?[A-Za-z0-9_-]+\.tar\.gz"
    return (isinstance(plan, dict) and plan.get("version") == 1
            and all(isinstance(plan.get(k), list) and all(isinstance(p, str) and re.fullmatch(path, p)
                                                           for p in plan[k]) for k in ("eligible", "protected")))


def reclaim_completed_checkpoints(run, run_gcs, gcs, log, keep=2, plan=None, timeout=120,
                                  stopping=lambda: False):
    """Bound the current run's acknowledged checkpoint replicas after writeback.

    Keep the newest two batch numbers for every member, including aliases such
    as final. Unacknowledged/in-progress files are never candidates. Only exact
    immutable GCS replicas can be evicted; recovery can fetch them on demand.
    The caller serializes this with run-state writeback, whose window is bounded
    (sync_run waits 330 s), so hashing stops at the deadline and resumes next time.
    Without a plan (rank 0) it is built from the ledger; other ranks pass rank 0's.
    """
    run = Path(run).resolve()
    deadline = time.monotonic() + timeout
    removed = total = 0

    def check_deadline():
        if stopping() or time.monotonic() >= deadline:
            raise TimeoutError("completed checkpoint cleanup stopped or exceeded its time budget")

    try:
        plan = checkpoint_plan(run, keep) if plan is None else plan
        if not valid_plan(plan):
            raise ValueError("invalid checkpoint plan")
        directory = private_path(run, run / "checkpoints")
        candidates = []
        for relative in sorted(set(plan["eligible"]) - set(plan["protected"])):
            path = private_path(run, directory / relative)
            if path.is_file():
                candidates.append((path, relative, file_identity(run, path)))
        if not candidates:
            return dict(files=0, bytes=0)
        prefix = run_gcs.rstrip("/") + "/checkpoints"
        objects = {obj.relative: obj for obj in gcs.list(prefix, allow_empty=True)}
        verified = []
        for p, rel, identity in candidates:
            check_deadline()
            if rel in objects and valid_file(p, objects[rel]):
                verified.append((p, objects[rel], identity))
        check_deadline()
        latest = {obj.relative: obj for obj in gcs.list(prefix, allow_empty=True)}
        for path, obj, identity in verified:
            if latest.get(obj.relative) != obj or file_identity(run, path) != identity:
                continue
            path.unlink()
            removed += 1
            total += obj.size
            emit(log, "completed_checkpoint_replica_reclaimed", local_path=str(path),
                 gcs=obj.uri, generation=obj.generation, bytes=obj.size)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        # TimeoutError is an OSError: the pass ends at the deadline and the next writeback continues.
        emit(log, "completed_checkpoint_cleanup_skipped", reason=str(exc))
    return dict(files=removed, bytes=total)


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
                      task_id=os.environ.get("SKYPILOT_TASK_ID", ""),
                      # storage_reconcile.py may remove a finished inference-only run whole (no durable state).
                      inference_only=bool(config.inference_only))
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


def process_matches_run(env, args, task_id, run_id, run):
    """A reused Ray namespace does not identify a retired managed-job attempt."""
    process_task = env.get("SKYPILOT_TASK_ID")
    return bool(
        (task_id and process_task == task_id)
        or env.get("TTD_RUN_DIR") == str(run) + "/client"
        or any(str(run) in arg for arg in args)
        # Preserve namespace-only orphans when attempt identity is unavailable.
        # Explicit paths above still protect users of the old root, even when
        # their managed-job identity differs (for example, a transfer process).
        or (env.get("RAY_NAMESPACE") == run_id and not process_task)
    )


def run_is_active(record, run):
    """An orphan can outlive bootstrap's lock. Unknown process access is unsafe."""
    import psutil
    for process in psutil.process_iter():
        try:
            if process.uids().real != os.getuid() or process.status() == psutil.STATUS_ZOMBIE:
                continue
            env = process.environ()
            if process_matches_run(env, process.cmdline(), record["task_id"],
                                   record["run_id"], run):
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
