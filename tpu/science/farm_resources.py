"""Host CPU partition and AC2 admission shared by trainer hosts and farms.

The science families keep their own contracts (``routing_resources`` takes
the NUMA-balanced top block, ``placement_resources`` the top ``4 * slots``
CPUs). AC2 takes the next ``slots * cpus`` CPUs below whichever science block
is present, and every service process (trainer, engine, ingress) is confined
to what remains. The partition is a pure function of the host affinity and
the declared families, so the Ray task that admits a candidate and the
service processes agree on the same disjoint sets.

Admission is a host-wide flock in ``/tmp/science-cpu-locks-<uid>``, like the
science graders, so two runtimes on one host can never oversubscribe a slot.
"""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import stat
import time

VERSION = 'farm-ac2-v1'
SERVICE_MIN = 24


def _lock(path, mode):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    handle = os.fdopen(fd, 'r+')
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        handle.close()
        raise RuntimeError('invalid grading admission lock')
    try:
        fcntl.flock(fd, mode | fcntl.LOCK_NB)
    except BaseException:
        handle.close()
        raise
    return handle


def _lock_root(root):
    root = Path(root) if root else Path('/tmp') / f'science-cpu-locks-{os.getuid()}'
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('grading locks must be private and owned by this user')
    return root


def partition(families, affinity=None, topology_root='/sys/devices/system/node'):
    """Return ({family: sorted grading CPUs}, service CPUs) for this host."""
    allowed = sorted(os.sched_getaffinity(0) if affinity is None else affinity)
    remaining = list(allowed)
    blocks = {}
    if 'routing' in families and 'placement' in families:
        raise ValueError('routing and placement grading cannot share one host partition')
    if 'placement' in families:
        slots = families['placement']['slots_per_host']
        count = 4 * slots
        if len(remaining) < count + 32:
            raise RuntimeError('placement slots require four CPUs each plus 32 service CPUs')
        blocks['placement'] = remaining[-count:]
        remaining = remaining[:-count]
    elif 'routing' in families:
        from .routing_resources import host_partition
        grading, service = host_partition(affinity=set(remaining), topology_root=topology_root)
        blocks['routing'] = sorted(grading)
        remaining = sorted(service)
    if 'math' in families:
        spec = families['math']
        count = spec['slots_per_host'] * spec['cpus']
        if len(remaining) - count < SERVICE_MIN:
            raise RuntimeError(f'math grading needs {count} CPUs plus {SERVICE_MIN} service CPUs')
        blocks['math'] = remaining[-count:]
        remaining = remaining[:-count]
    if families and len(remaining) < SERVICE_MIN:
        raise RuntimeError('grading partition leaves too few service CPUs')
    return blocks, remaining


def math_slot_cpus(families, slot, affinity=None):
    spec = families['math']
    block, _ = partition(families, affinity)
    cpus = block['math']
    if not 0 <= slot < spec['slots_per_host']:
        raise ValueError('invalid ac2 grading slot')
    return cpus[slot * spec['cpus']:(slot + 1) * spec['cpus']]


def acquire_math(families, deadline_seconds=1100, root=None, affinity=None):
    """Hold one AC2 slot; returns (slot, cpus, ExitStack releasing the lock).

    The CPU map is pinned per host in ``farm-ac2-map.json`` so two runtimes
    with different affinities cannot silently disagree about slot CPU sets.
    """
    spec = families['math']
    slots = spec['slots_per_host']
    block, _ = partition(families, affinity)
    cpus = block['math']
    root = _lock_root(root)
    deadline = time.monotonic() + deadline_seconds
    while True:
        stack = ExitStack()
        try:
            with _lock(root / f'{VERSION}-map.lock', fcntl.LOCK_EX):
                mapping = root / f'{VERSION}-map.json'
                active = []
                try:
                    for slot in range(slots):
                        try:
                            active.append(_lock(root / f'{VERSION}-{slot}.lock', fcntl.LOCK_EX))
                        except BlockingIOError:
                            break
                    if len(active) != slots:
                        if json.loads(mapping.read_text()) != cpus:
                            raise RuntimeError('active ac2 grading jobs use a different CPU map')
                    else:
                        mapping.write_text(json.dumps(cpus))
                finally:
                    for handle in active:
                        handle.close()
                for slot in range(slots):
                    try:
                        handle = _lock(root / f'{VERSION}-{slot}.lock', fcntl.LOCK_EX)
                    except BlockingIOError:
                        continue
                    stack.enter_context(handle)
                    return slot, cpus[slot * spec['cpus']:(slot + 1) * spec['cpus']], stack
        except BlockingIOError:
            pass
        except BaseException:
            stack.close()
            raise
        stack.close()
        if time.monotonic() >= deadline:
            raise TimeoutError('ac2 grading admission timed out before candidate execution')
        time.sleep(min(.2, max(0, deadline - time.monotonic())))
