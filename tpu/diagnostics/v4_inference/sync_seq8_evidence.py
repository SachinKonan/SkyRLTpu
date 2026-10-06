"""Additive one-minute writeback for the bounded job396 diagnostic artifacts."""
import json
import os
from pathlib import Path
import subprocess
import time


def main():
    root = Path.home() / ".cache/skyrl-ray"
    source = root / "runs/qwen-ray-v4-64-concurrency-001/validation/seq8-20260907"
    target = "gs://sk7524-tinker-tpu-us-central2/ray-training/qwen-ray-v4-64-concurrency-001/validation/"
    env = dict(os.environ, CLOUDSDK_STORAGE_PROCESS_COUNT="2", CLOUDSDK_STORAGE_THREAD_COUNT="4")
    deadline = time.monotonic() + 4 * 3600
    while time.monotonic() < deadline:
        try:
            alive = b"run_seq8_job396.py" in Path("/proc/422992/cmdline").read_bytes()
        except FileNotFoundError:
            alive = False
        result = subprocess.run(["gcloud", "storage", "cp", "--recursive", str(source), target],
                                env=env, capture_output=True, text=True, timeout=300)
        print(json.dumps(dict(time=time.time(), returncode=result.returncode, runner_alive=alive,
                              detail=result.stderr[-2000:])), flush=True)
        if not alive and result.returncode == 0:
            return
        time.sleep(60)
    raise TimeoutError("evidence writeback deadline exceeded")


if __name__ == "__main__":
    main()
