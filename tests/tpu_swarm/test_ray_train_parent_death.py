import subprocess
import sys
import time

import psutil


def test_actor_death_cleans_owned_subprocess_without_touching_peer(tmp_path):
    log = tmp_path / "child.log"
    code = (
        "from pathlib import Path; import sys,time; "
        "from tpu.swarm.ray_train.process import Process; "
        "p=Process([sys.executable,'-u','-c',"
        "'import os,time; print(os.getpid(),flush=True); time.sleep(120)'],"
        f"Path({str(log)!r}),term_grace=1); time.sleep(120)"
    )
    parent = subprocess.Popen([sys.executable, "-c", code])
    peer = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    child = None
    try:
        deadline = time.monotonic() + 10
        while (not log.exists() or not log.read_text().strip()) and time.monotonic() < deadline:
            time.sleep(0.1)
        child = psutil.Process(int(log.read_text().strip()))
        parent.kill()
        parent.wait(timeout=5)
        deadline = time.monotonic() + 10
        while child.is_running() and child.status() != psutil.STATUS_ZOMBIE and time.monotonic() < deadline:
            time.sleep(0.1)
        assert not child.is_running() or child.status() == psutil.STATUS_ZOMBIE
        assert peer.poll() is None
    finally:
        if parent.poll() is None:
            parent.kill()
            parent.wait(timeout=5)
        peer.terminate()
        peer.wait(timeout=5)
        if child and child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
            child.kill()
