"""Scoped canary cleanup, including its descendants; no global ray stop."""
import os
import psutil
import time


def wait_owned(processes, timeout):
    # Some TPU kernels reject pidfd_open used by psutil.wait_procs. Checking
    # Process identity/status also avoids signaling a reused PID or a zombie.
    deadline = time.monotonic() + timeout
    alive = list(processes)
    while alive:
        remaining = []
        for process in alive:
            try:
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    remaining.append(process)
            except psutil.NoSuchProcess:
                pass
        alive = remaining
        if not alive or time.monotonic() >= deadline:
            break
        time.sleep(min(0.1, max(0, deadline - time.monotonic())))
    return alive


def stop_owned(victims):
    for process in victims:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    alive = wait_owned(victims, timeout=15)
    for process in alive:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    remaining = wait_owned(alive, timeout=5)
    if remaining:
        raise RuntimeError(f"canary processes did not stop: {[p.pid for p in remaining]}")


def main():
    ray_tmp = os.environ["RAY_TMPDIR"].rstrip("/")
    root = os.environ["CANARY_ROOT"].rstrip("/")
    address = os.environ["CANARY_RAY_ADDRESS"]
    victims = {}
    for p in psutil.process_iter(["pid", "cmdline"]):
        if p.pid == os.getpid():
            continue
        args = p.info["cmdline"] or []
        cmd = " ".join(args)
        owned_download = any(a.endswith("/gcloud.py") for a in args) and any(
            a.startswith(root + "/") for a in args)
        if f"--gcs-address={address}" in cmd or f"{ray_tmp}/" in cmd or owned_download:
            try:
                for child in p.children(recursive=True):
                    victims[child.pid] = child
                victims[p.pid] = p
            except psutil.NoSuchProcess:
                pass
    stop_owned(list(victims.values()))


if __name__ == "__main__":
    main()
