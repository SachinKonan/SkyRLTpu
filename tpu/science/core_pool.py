"""One host-wide grading core pool shared by every grading family.

The runtime reserves a fixed service block (engines, trainer, ingress, Ray)
and hands every other core to grading. A grade asks for a number of cores and
GiB, never for a family slot: under a short host-wide lock it takes free
per-core locks (one NUMA node when possible) and per-GiB memory tokens, and
holds them for its lifetime. A systemd unit then pins the program to exactly
those cores and caps its memory, so each program keeps the same contract
(e.g. AC2 2 cores/4 GiB, placement case 4/4, routing program 10/20) whatever
else the host grades. Ray admits by the same counts (its CPU and memory
budget is this pool), so families mix freely without oversubscription.

All locks are ``flock``s in ``/tmp/science-cpu-locks-<uid>``: a dead holder
releases its cores and memory automatically.
"""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import random
import stat
import time

VERSION = 'core-pool-v1'
POOL_FILE = VERSION + '.json'
SERVICE_MIN = 24


class PoolUnavailable(RuntimeError):
    """The host runtime has not installed a grading core pool."""


def lock_root(root=None):
    root = Path(root) if root else Path('/tmp') / f'science-cpu-locks-{os.getuid()}'
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('grading locks must be private and owned by this user')
    return root


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


def _parse_cpus(text):
    result = set()
    for part in text.strip().split(','):
        if part:
            ends = list(map(int, part.split('-')))
            result.update(range(ends[0], ends[-1] + 1))
    return result


def numa_nodes(allowed, topology_root='/sys/devices/system/node'):
    """{cpu: node} over the allowed CPUs (one node when topology is absent)."""
    allowed = set(allowed)
    mapping = {}
    for index, path in enumerate(sorted(Path(topology_root).glob('node*/cpulist'))):
        for cpu in _parse_cpus(path.read_text()) & allowed:
            mapping[cpu] = index
    if set(mapping) != allowed:
        return {cpu: 0 for cpu in allowed}
    return mapping


def layout(service_cpus, affinity=None, topology_root='/sys/devices/system/node'):
    """Return (grading CPUs, service CPUs, {cpu: node}) for this host.

    Services take the lowest CPUs of every NUMA node in turn, so both nodes
    keep grading capacity and service work stays spread out.
    """
    allowed = sorted(os.sched_getaffinity(0) if affinity is None else affinity)
    if type(service_cpus) is not int or service_cpus < SERVICE_MIN:
        raise ValueError(f'the service block needs at least {SERVICE_MIN} CPUs')
    if len(allowed) < service_cpus + 2:
        raise RuntimeError(f'{len(allowed)} CPUs cannot hold {service_cpus} service CPUs and a grading pool')
    nodes = numa_nodes(allowed, topology_root)
    groups = [sorted(c for c in allowed if nodes[c] == n) for n in sorted(set(nodes.values()))]
    service = []
    while len(service) < service_cpus:
        for group in groups:
            if group and len(service) < service_cpus:
                service.append(group.pop(0))
    service = sorted(service)
    grading = sorted(set(allowed) - set(service))
    return grading, service, {cpu: nodes[cpu] for cpu in grading}


def install(service_cpus, memory_gib, *, root=None, affinity=None, topology_root='/sys/devices/system/node'):
    """Publish this host's pool for graders; refuse to change a pool in use."""
    if type(memory_gib) is not int or memory_gib < 1:
        raise ValueError('grading memory budget must be a positive integer of GiB')
    grading, service, nodes = layout(service_cpus, affinity, topology_root)
    document = dict(version=VERSION, grading=grading, service=service,
                    nodes={str(c): n for c, n in nodes.items()}, memory_gib=memory_gib)
    root = lock_root(root)
    path = root / POOL_FILE
    with _blocking(root / (VERSION + '.lock')):
        if path.is_file():
            current = json.loads(path.read_text())
            if current != document and _held(root, current):
                raise RuntimeError('an active grading job uses a different core pool; drain it first')
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(document))
        temporary.replace(path)
    return document


def read(root=None):
    path = lock_root(root) / POOL_FILE
    if not path.is_file():
        raise PoolUnavailable('this host has no grading core pool installed')
    document = json.loads(path.read_text())
    if document.get('version') != VERSION:
        raise PoolUnavailable('grading core pool version mismatch')
    return document


class _blocking:
    """Short exclusive host-wide lock serializing pool decisions."""
    def __init__(self, path):
        self.path = path
        self.handle = None

    def __enter__(self):
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        self.handle = os.fdopen(fd, 'r+')
        fcntl.flock(fd, fcntl.LOCK_EX)
        return self

    def __exit__(self, *exc):
        self.handle.close()


def _held(root, document):
    """True when any grade still holds a core of ``document``."""
    for cpu in document.get('grading', []):
        try:
            _lock(root / f'{VERSION}-cpu-{cpu}.lock', fcntl.LOCK_EX).close()
        except BlockingIOError:
            return True
    return False


def _choose(free, count, nodes):
    """Prefer one NUMA node (the fullest one that fits); else span nodes."""
    by_node = {}
    for cpu in free:
        by_node.setdefault(nodes[cpu], []).append(cpu)
    fitting = [cpus for cpus in by_node.values() if len(cpus) >= count]
    if fitting:
        return sorted(max(fitting, key=len))[:count]
    chosen = []
    for cpus in sorted(by_node.values(), key=len, reverse=True):
        chosen.extend(sorted(cpus)[:count - len(chosen)])
        if len(chosen) == count:
            return sorted(chosen)
    return None


def try_acquire(cpus, memory_gib, *, root=None):
    """One attempt: ([cores], ExitStack holding them) or None if busy."""
    root = lock_root(root)
    document = read(root)
    if type(cpus) is not int or not 1 <= cpus <= len(document['grading']):
        raise ValueError(f'a grade cannot hold {cpus} of {len(document["grading"])} pool CPUs')
    if type(memory_gib) is not int or not 1 <= memory_gib <= document['memory_gib']:
        raise ValueError(f'a grade cannot hold {memory_gib} of {document["memory_gib"]} GiB')
    nodes = {int(c): n for c, n in document['nodes'].items()}
    stack = ExitStack()
    try:
        with _blocking(root / (VERSION + '.lock')):
            probes = {}
            try:
                for cpu in document['grading']:
                    try:
                        probes[cpu] = _lock(root / f'{VERSION}-cpu-{cpu}.lock', fcntl.LOCK_EX)
                    except BlockingIOError:
                        continue
                chosen = _choose(list(probes), cpus, nodes)
                if chosen is None:
                    return None
                for cpu in chosen:
                    stack.enter_context(probes.pop(cpu))
            finally:
                for handle in probes.values():
                    handle.close()
            held = 0
            for token in range(document['memory_gib']):
                if held == memory_gib:
                    break
                try:
                    stack.enter_context(_lock(root / f'{VERSION}-gib-{token}.lock', fcntl.LOCK_EX))
                    held += 1
                except BlockingIOError:
                    continue
            if held < memory_gib:
                stack.close()
                return None
        return chosen, stack
    except BaseException:
        stack.close()
        raise


def acquire(cpus, memory_gib, *, deadline_seconds=1100, root=None):
    """Block until ``cpus`` cores and ``memory_gib`` GiB are free; raise TimeoutError."""
    deadline = time.monotonic() + deadline_seconds
    while True:
        held = try_acquire(cpus, memory_gib, root=root)
        if held is not None:
            return held
        if time.monotonic() >= deadline:
            raise TimeoutError(f'grading pool admission timed out waiting for {cpus} CPUs and {memory_gib} GiB')
        time.sleep(min(.2 + random.random() * .3, max(0, deadline - time.monotonic())))
