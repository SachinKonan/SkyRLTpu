"""Stop explicitly retired task processes, never infer retirement from idle TPU state."""
import os
import hashlib
import json
import time
from pathlib import Path

import psutil


def workload_kind(args):
    if any(token in args for token in ("skyrl.tinker.api", "skyrl.tinker.engine", "skyrl.backends.rpc")):
        return "trainer"
    if any(arg.endswith("/vllm_tpu_server.py") or arg.startswith("VLLM::") for arg in args):
        return "inference"
    if any(Path(arg).name in ("cell_worker.sh", "cell_monitor.sh") or
           (Path(arg).name.startswith("sidecar_") and arg.endswith(".sh")) for arg in args):
        return "cell supervisor"
    return None


def process_identity(process):
    return {"pid": process.pid, "created": process.create_time(),
            "command_sha256": hashlib.sha256(json.dumps(process.cmdline()).encode()).hexdigest()}


def wait_for_exit(processes, timeout):
    # Ray's bundled psutil can use pidfd_open, unsupported on some TPU kernels.
    # Status polling also treats non-child zombies as stopped without reaping.
    pending = list(processes)
    deadline = time.monotonic() + timeout
    while pending:
        alive = []
        for process in pending:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    alive.append(process)
            except psutil.NoSuchProcess:
                pass
        pending = alive
        if not pending or time.monotonic() >= deadline:
            return pending
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    return []


def retire_workloads(retired_task_ids, audited=None):
    """Caller must verify these immutable SkyPilot task IDs are terminal first.

    Run only inside an allocated managed job, before any new workload starts.
    No generic tmux-session kill: a session name can have been reused already.
    """
    retired = set(retired_task_ids)
    audited = audited or {}
    identities = audited.get("processes", [])
    if identities and audited.get("boot_id") != Path("/proc/sys/kernel/random/boot_id").read_text().strip():
        identities = []
    roots = []
    unidentified = []
    for process in psutil.process_iter(["pid", "cmdline"]):
        kind = workload_kind(process.info["cmdline"] or [])
        if not kind:
            continue
        try:
            if process.status() == psutil.STATUS_ZOMBIE:
                continue
            task_id = process.environ().get("SKYPILOT_TASK_ID")
            identified = bool(task_id and task_id in retired) or (
                bool(identities) and process_identity(process) in identities)
            if process.uids().real != os.getuid():
                raise RuntimeError(f"existing {kind} PID {process.pid} is not explicitly retired")
            if identified:
                roots.append(process)
            else:
                unidentified.append((process, kind))
        except psutil.NoSuchProcess:
            pass
    # Validate every root before sending any signal. psutil checks PID reuse.
    selected = {}
    for root in roots:
        try:
            selected[root.pid] = root
            selected.update({p.pid: p for p in root.children(recursive=True)})
        except psutil.NoSuchProcess:
            pass
    # An engine child is covered by its audited parent even when it did not
    # inherit the task ID. A reparented engine needs its own exact authorization.
    for process, kind in unidentified:
        if process.pid not in selected:
            raise RuntimeError(f"existing {kind} PID {process.pid} is not explicitly retired")
    for process in selected.values():
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    alive = wait_for_exit(selected.values(), timeout=20)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    alive = wait_for_exit(alive, timeout=5)
    if any(p.is_running() and p.status() != psutil.STATUS_ZOMBIE for p in alive):
        raise RuntimeError("retired workload did not stop")
    return sorted(selected)
