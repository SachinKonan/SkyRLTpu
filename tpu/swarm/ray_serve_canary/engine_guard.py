"""Stop this engine's process group if its owning Serve replica disappears."""
import os
import signal
import subprocess
import sys
import time

parent = int(sys.argv[1])
child = subprocess.Popen(sys.argv[2:], start_new_session=True)


def stop(*_):
    try:
        os.killpg(child.pid, signal.SIGTERM)
        child.wait(timeout=15)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
    except ProcessLookupError:
        pass


signal.signal(signal.SIGTERM, lambda *_: (stop(), sys.exit(143)))
signal.signal(signal.SIGINT, lambda *_: (stop(), sys.exit(130)))
try:
    while child.poll() is None:
        if os.getppid() != parent:
            stop()
            sys.exit(1)
        time.sleep(1)
    sys.exit(child.returncode)
finally:
    stop()
