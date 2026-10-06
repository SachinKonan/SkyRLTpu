"""Host-wide CPU grading admission, independent of a run's payload directory."""
import fcntl
import os
from pathlib import Path
import stat
import time


def validate_slots(slots):
    if type(slots) is not int or not 1 <= slots <= 16:
        raise ValueError('CPU grading slots must be in [1,16]; each reserves 8 GiB')
    return slots


def slot_cpus(slot):
    if type(slot) is not int or not 0 <= slot < 16:
        raise ValueError('invalid CPU grading slot')
    return list(range(16 + slot * 4, 20 + slot * 4))


def acquire_slot(root=None, slots=2, deadline_seconds=2400):
    validate_slots(slots)
    root = Path(root) if root is not None else Path('/tmp') / f'science-cpu-locks-{os.getuid()}'
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('CPU slot directory must be private and owned by the worker')
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        for slot in range(slots):
            fd = os.open(root / f'cpu-slot-{slot}.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
            handle = os.fdopen(fd, 'r+')
            try:
                info = os.fstat(fd)
                if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
                    raise RuntimeError('invalid CPU grading slot lock')
                try:
                    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                return slot, handle
            except BaseException:
                handle.close()
                raise
        time.sleep(min(.2, max(0, deadline - time.monotonic())))
    raise TimeoutError('CPU admission timed out before candidate execution')
