"""Versioned, opt-in routing resource contract and host-wide admission."""
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import stat
import time

VERSION = 'parallel-v2'
CASE_WORKERS = 4
CASE_CPUS = 2
CASE_GIB = 4
# Conservative packing includes two CPUs/four GiB for coordinator and build.
PROGRAM_CPUS = 10
PROGRAM_GIB = 20
MAX_PROGRAMS = 10
HOST_CPUS = 100
HOST_GIB = 200
CANDIDATE_SECONDS = 1900
OUTER_SECONDS = 2100
GEMINI_TARGETS = {'q20': 13470, 'willow': 31481, 'heron_fez': 42396}


def contract():
    return dict(version=VERSION, case_workers=CASE_WORKERS, case_cpus=CASE_CPUS,
                case_memory_gib=CASE_GIB, program_cpus=PROGRAM_CPUS,
                program_memory_gib=PROGRAM_GIB, max_programs=MAX_PROGRAMS,
                host_cpus=HOST_CPUS, host_memory_gib=HOST_GIB,
                candidate_seconds=CANDIDATE_SECONDS, outer_seconds=OUTER_SECONDS,
                layout_trials=20, routing_trials=20, gemini_targets=GEMINI_TARGETS)


def validate_request(value):
    if value != contract():
        raise ValueError('routing resource contract mismatch')


def parse_cpus(text):
    result = set()
    for part in text.strip().split(','):
        if not part: continue
        ends = list(map(int, part.split('-')))
        result.update(range(ends[0], ends[-1]+1))
    return sorted(result)


def host_partition(affinity=None, topology_root='/sys/devices/system/node'):
    """Reserve 100 grading CPUs, balanced over actual NUMA nodes.

    At least 24 other CPUs remain for trainer/inference and service overhead.
    The returned order keeps each program's workers on predictable CPU sets.
    """
    allowed = set(os.sched_getaffinity(0) if affinity is None else affinity)
    if len(allowed) < HOST_CPUS + 24:
        raise RuntimeError('parallel routing needs 100 grading CPUs plus 24 service CPUs')
    groups = [sorted(set(parse_cpus(p.read_text())) & allowed)
              for p in sorted(Path(topology_root).glob('node*/cpulist'))]
    groups = [g for g in groups if g] or [sorted(allowed)]
    if set().union(*map(set, groups)) != allowed:
        raise RuntimeError('NUMA map does not cover host affinity')
    # Select the upper portion of each node, leaving service CPUs on every node.
    selected = []
    while len(selected) < HOST_CPUS:
        for group in groups:
            if group and len(selected) < HOST_CPUS: selected.append(group.pop())
    return sorted(selected), sorted(allowed - set(selected))


def _lock(path, mode):
    fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    handle = os.fdopen(fd, 'r+')
    info = os.fstat(fd)
    if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_nlink != 1:
        handle.close(); raise RuntimeError('invalid routing admission lock')
    try: fcntl.flock(fd, mode | fcntl.LOCK_NB)
    except BaseException: handle.close(); raise
    return handle


def acquire(slots=MAX_PROGRAMS, deadline_seconds=2400, root=None, affinity=None):
    if type(slots) is not int or not 1 <= slots <= MAX_PROGRAMS:
        raise ValueError('parallel routing supports at most ten programs per VM')
    cpus, _ = host_partition(affinity)
    root = Path(root) if root else Path('/tmp') / f'science-cpu-locks-{os.getuid()}'
    root.mkdir(mode=0o700, exist_ok=True)
    info = root.lstat()
    if not stat.S_ISDIR(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('CPU slot directory must be private and owned by the worker')
    deadline = time.monotonic() + deadline_seconds
    while time.monotonic() < deadline:
        stack = ExitStack()
        try:
            # Existing graders use exclusive locks here. Shared holds on ALL old
            # slots prevent either implementation from oversubscribing the other.
            for slot in range(16):
                stack.enter_context(_lock(root / f'cpu-slot-{slot}.lock', fcntl.LOCK_SH))
            # Pin the CPU map across independent roots/restricted caller affinities.
            with _lock(root / 'parallel-map.lock', fcntl.LOCK_EX):
                active = []
                try:
                    for slot in range(MAX_PROGRAMS):
                        try: active.append(_lock(root / f'{VERSION}-{slot}.lock', fcntl.LOCK_EX))
                        except BlockingIOError: break
                    mapping = root / 'parallel-map.json'
                    if len(active) != MAX_PROGRAMS:
                        if json.loads(mapping.read_text()) != cpus:
                            raise RuntimeError('active routing jobs use a different CPU map')
                    else:
                        mapping.write_text(json.dumps(cpus))
                finally:
                    for handle in active: handle.close()
                for slot in range(slots):
                    try: handle = _lock(root / f'{VERSION}-{slot}.lock', fcntl.LOCK_EX)
                    except BlockingIOError: continue
                    stack.enter_context(handle)
                    return slot, cpus[slot*PROGRAM_CPUS:(slot+1)*PROGRAM_CPUS], stack
        except BlockingIOError:
            pass
        except BaseException:
            stack.close(); raise
        stack.close()
        time.sleep(min(.2, max(0, deadline-time.monotonic())))
    raise TimeoutError('parallel routing admission timed out before evaluation')
