"""Owned process groups with parent-death cleanup and bounded waits."""
from __future__ import annotations

import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


class Process:
    def __init__(self, command, log: Path, env=None, cwd=None, term_grace=15):
        log.parent.mkdir(parents=True, exist_ok=True)
        self.log = log
        self.log_offset = log.stat().st_size if log.exists() else 0
        self.output = log.open("ab", buffering=0)
        self.term_grace = term_grace
        guardian = [sys.executable, str(Path(__file__).resolve()), "guard", str(os.getpid()),
                    json.dumps(list(command)), str(term_grace)]
        self.process = subprocess.Popen(guardian, stdout=self.output, stderr=subprocess.STDOUT,
                                        env=env, cwd=cwd, start_new_session=True)

    def poll(self):
        return self.process.poll()

    def stop(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=self.term_grace + 10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        self.output.close()


def guard(parent, command, term_grace=15):
    child = subprocess.Popen(command, start_new_session=True)
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        while child.poll() is None:
            if stopping or os.getppid() != parent:
                return 143
            time.sleep(0.2)
        return child.returncode
    finally:
        # Descendants may survive a driver exit. Always signal the private group.
        try:
            os.killpg(child.pid, signal.SIGTERM)
            deadline = time.monotonic() + term_grace
            while time.monotonic() < deadline:
                child.poll()
                try:
                    os.killpg(child.pid, 0)
                except ProcessLookupError:
                    break
                time.sleep(0.1)
            else:
                os.killpg(child.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        child.wait(timeout=5)


if __name__ == "__main__":
    if sys.argv[1] != "guard":
        raise SystemExit("internal guardian entrypoint")
    raise SystemExit(guard(int(sys.argv[2]), json.loads(sys.argv[3]), float(sys.argv[4])))
