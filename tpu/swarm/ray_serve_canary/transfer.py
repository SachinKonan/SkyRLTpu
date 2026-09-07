"""Measure native gcloud transfers and terminate their process group on exit."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time


def disk_snapshot(path):
    device = path.stat().st_dev
    stat = Path(f"/sys/dev/block/{os.major(device)}:{os.minor(device)}/stat")
    if not stat.exists():
        return None
    fields = [int(x) for x in stat.read_text().split()]
    return {"write_bytes": fields[6] * 512, "write_ios": fields[4], "busy_ms": fields[9]}


def rates(before, after, seconds):
    if before is None or after is None:
        return {}
    return {"disk_write_mib_s": (after["write_bytes"] - before["write_bytes"]) / 2**20 / seconds,
            "disk_write_iops": (after["write_ios"] - before["write_ios"]) / seconds,
            "disk_busy_fraction": (after["busy_ms"] - before["busy_ms"]) / 1000 / seconds}


def run_transfer(command, root, destination, label, stdin_text="", check=True):
    root, destination = Path(root), Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    metrics = root / f"transfer-{label}.jsonl"
    started = time.monotonic()
    last_time, last_disk = started, disk_snapshot(destination)

    def report(event, **extra):
        nonlocal last_time, last_disk
        now, disk = time.monotonic(), disk_snapshot(destination)
        sizes = []
        for p in destination.rglob("*"):
            try:
                if p.is_file():
                    sizes.append(p.stat().st_size)
            except FileNotFoundError:
                pass
        row = {"event": event, "time": time.time(), "elapsed_s": now - started,
               "process_count": os.environ.get("CLOUDSDK_STORAGE_PROCESS_COUNT"),
               "thread_count": os.environ.get("CLOUDSDK_STORAGE_THREAD_COUNT"),
               "slice_threshold": os.environ.get("CLOUDSDK_STORAGE_SLICED_OBJECT_DOWNLOAD_THRESHOLD"),
               "destination_files": len(sizes), "destination_logical_bytes": sum(sizes),
               **rates(last_disk, disk, max(now - last_time, 0.001)), **extra}
        with metrics.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)
        last_time, last_disk = now, disk

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    previous = {s: signal.signal(s, interrupted) for s in (signal.SIGTERM, signal.SIGINT)}
    proc = None
    try:
        with tempfile.TemporaryFile(mode="w+") as inputs:
            inputs.write(stdin_text)
            inputs.seek(0)
            with (root / f"transfer-{label}.log").open("a") as log:
                proc = subprocess.Popen(command, stdin=inputs, stdout=log, stderr=subprocess.STDOUT,
                                        start_new_session=True)
                report("transfer_start", pid=proc.pid)
                while True:
                    try:
                        rc = proc.wait(timeout=30)
                        break
                    except subprocess.TimeoutExpired:
                        report("transfer_progress")
                report("transfer_end", returncode=rc)
        if check and rc:
            raise subprocess.CalledProcessError(rc, command)
        return rc
    finally:
        if proc is not None:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait(timeout=10)
            except ProcessLookupError:
                pass
        for signum, handler in previous.items():
            signal.signal(signum, handler)


if __name__ == "__main__":
    run_transfer(sys.argv[3:], Path(os.environ["CANARY_ROOT"]), Path(sys.argv[2]), sys.argv[1])
