"""Tear the executor's Ray down even when the bootstrap dies by SIGKILL.

SkyPilot's cancel ends the task with a hard kill after a short grace, which
skips the bootstrap's `finally: stop_ray(...)`; the head's gcs_server then
keeps the executor's Ray port and the next job on that worker fails its
port check (jobs 548, 612/614 -> 613). The bootstrap spawns this module in
its own session right after `ray start`; it waits for the bootstrap PID to
exit and then terminates the Ray daemons that (a) reference this run's
Ray temp dir and (b) were created before the bootstrap exited, so a newer
job's Ray on the same worker is left alone.
"""
from __future__ import annotations

import os
import sys
import time


def matches(cmdline, ray_tmp: str) -> bool:
    return any(ray_tmp + "/" in arg or arg == ray_tmp for arg in cmdline or [])


def select_victims(processes, ray_tmp: str, before: float, self_pid: int):
    """Processes (with children) whose command line names `ray_tmp` and that
    started before `before`; `processes` yields objects with pid, cmdline(),
    create_time(), children(recursive=True)."""
    selected = {}
    for process in processes:
        try:
            if process.pid == self_pid or not matches(process.cmdline(), ray_tmp):
                continue
            if process.create_time() >= before:
                continue
            selected[process.pid] = process
            for child in process.children(recursive=True):
                if child.create_time() < before:
                    selected[child.pid] = child
        except Exception:  # process vanished or is unreadable
            continue
    return selected


def reap(ray_tmp: str, before: float, log):
    import psutil
    victims = select_victims(psutil.process_iter(), ray_tmp, before, os.getpid())
    for process in victims.values():
        try:
            process.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(list(victims.values()), timeout=20)
    for process in alive:
        try:
            process.kill()
        except psutil.Error:
            pass
    psutil.wait_procs(alive, timeout=10)
    print(f"{time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} reaped {sorted(victims)}", file=log, flush=True)


def main():
    ray_tmp, parent = sys.argv[1], int(sys.argv[2])
    import psutil
    try:
        target = psutil.Process(parent)
        start = target.create_time()
        while True:
            try:
                if not target.is_running() or target.status() == psutil.STATUS_ZOMBIE or psutil.Process(parent).create_time() != start:
                    break
            except psutil.NoSuchProcess:
                break
            time.sleep(2)
    except psutil.NoSuchProcess:
        pass
    # Let a graceful shutdown finish its own stop_ray first.
    time.sleep(5)
    reap(ray_tmp, time.time(), sys.stdout)


if __name__ == "__main__":
    main()
