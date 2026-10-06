"""Structured per-host status, usable before Ray is initialized."""
import json
import os
from pathlib import Path
import socket
import time


def emit(path: Path, event: str, **fields):
    record = dict(time=time.time(), event=event, host=socket.gethostname(), pid=os.getpid(), **fields)
    text = json.dumps(record, sort_keys=True)
    print(text, flush=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as output:
        output.write(text + "\n")
    return record
