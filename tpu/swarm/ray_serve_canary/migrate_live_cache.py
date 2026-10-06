"""Move an active pre-engine canary restore to RAM without releasing its job."""
import os
from pathlib import Path
import signal
import time

import psutil

from ram_cache import setup


def alive(process):
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def main():
    root = Path(os.environ["CANARY_ROOT"]).resolve()
    supervisors = []
    for p in psutil.process_iter(["cmdline"]):
        args = p.info["cmdline"] or []
        if str(root / "source/tpu/vllm_tpu_server.py") in args:
            raise RuntimeError("refusing live cache migration after engine launch")
        if str(root / "code/prepare_cache.py") in args:
            supervisors.append(p)
    if len(supervisors) != 1:
        raise RuntimeError(f"expected one active cache supervisor, found {len(supervisors)}")
    supervisor = supervisors[0]
    supervisor.suspend()
    try:
        children = supervisor.children()
        downloads = [p for p in children if any(a.endswith('/gcloud.py') for a in p.cmdline())]
        if len(downloads) != 1:
            raise RuntimeError("cache supervisor is not waiting on exactly one gcloud batch")
        child = downloads[0]
        if os.getpgid(child.pid) != child.pid:
            raise RuntimeError("gcloud is not in its expected private process group")
        processes = [child, *child.children(recursive=True)]
        os.killpg(child.pid, signal.SIGTERM)
        for i in range(90):
            if not any(alive(p) for p in processes):
                break
            if i == 15:
                os.killpg(child.pid, signal.SIGKILL)
            time.sleep(1)
        else:
            raise RuntimeError("download processes did not stop; paths left unchanged")
        setup(root)
    finally:
        if alive(supervisor):
            supervisor.resume()


if __name__ == "__main__":
    main()
