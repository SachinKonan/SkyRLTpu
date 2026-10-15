"""No-sudo grading units: the limits ``sudo systemd-run`` puts on a grade, kept by a supervisor.

Graders normally run each grade as a transient systemd unit (AllowedCPUs, CPUQuota, MemoryMax with
MemorySwapMax=0, TasksMax, RuntimeMaxSec, KillMode=control-group, OOMPolicy=stop, PrivateTmp). Hosts
without sudo or a writable cgroup (della Slurm CPU workers) select ``SCIENCE_GRADING_ISOLATION=bwrap``
and run the same command under this supervisor instead, with the same numbers:

- CPUs: the unit starts pinned to its pool cores; a thread that widens its affinity is pinned back at
  the next sample (a cpuset would intersect it). Pinning also bounds CPU time like the quota.
- Memory: every descendant is sampled every 50 ms (the supervisor is their subreaper, so none
  escapes); anonymous + shmem + swap over the limit, confirmed by PSS, kills the tree (OOMPolicy=stop).
- Tasks: more threads than TasksMax kills the tree (a pids cgroup would refuse the fork instead).
- Time: at the deadline the tree is stopped (TERM, 2 s, KILL), like RuntimeMaxSec/TimeoutStopSec=2.
- Lifetime: the tree is stopped when the command exits (KillMode=control-group), when the
  supervisor is told to stop, or when its owner (the Ray worker) dies.
- PrivateTmp: the command runs in a bubblewrap mount namespace with fresh /tmp and /var/tmp.
Candidates keep their own bubblewrap sandbox (``isolation.command``), unchanged.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import pwd
import signal
import subprocess
import sys
import time

MODE = 'SCIENCE_GRADING_ISOLATION'
# Set only inside a supervised unit: the unit's memory limit (systemd-run passes no environment).
LIMIT = 'SCIENCE_UNIT_MEMORY_BYTES'
SAMPLE_SECONDS = .05
STOP_SECONDS = 2
PR_SET_CHILD_SUBREAPER = 36


def mode():
    value = os.environ.get(MODE) or 'systemd'
    if value not in ('systemd', 'bwrap'):
        raise ValueError(f'{MODE} must be systemd or bwrap')
    return value


def inside():
    """True in a process this supervisor started (cgroup_limits/routing_parallel use no cgroup then)."""
    return LIMIT in os.environ


def command(argv, *, cpus, memory_gib, seconds, tasks_max, private_tmp=False):
    """The supervisor command line that stands in for ``sudo systemd-run ... argv``."""
    return [sys.executable, '-m', 'tpu.science.unit', '--cpus', ','.join(map(str, cpus)),
            '--memory-gib', str(memory_gib), '--seconds', str(seconds), '--tasks-max', str(tasks_max),
            *(['--private-tmp'] if private_tmp else []), '--', *map(str, argv)]


def environment():
    """A fresh environment like a systemd service's (systemd-run passes none of the caller's)."""
    user = pwd.getpwuid(os.getuid())
    return {'PATH': '/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin', 'LANG': os.environ.get('LANG', 'C.UTF-8'),
            'HOME': user.pw_dir, 'USER': user.pw_name, 'LOGNAME': user.pw_name, 'SHELL': user.pw_shell,
            MODE: 'bwrap'}


def tree(roots):
    """Every live descendant of ``roots`` (roots included), through /proc children lists."""
    found, stack = [], list(roots)
    while stack:
        pid = stack.pop()
        found.append(pid)
        try:
            tids = os.listdir(f'/proc/{pid}/task')
        except OSError:
            continue
        for tid in tids:
            try:
                stack.extend(int(c) for c in Path(f'/proc/{pid}/task/{tid}/children').read_text().split())
            except OSError:
                pass
    return found


def _fields(path, names):
    values = {}
    try:
        for line in Path(path).read_text().splitlines():
            key, _, value = line.partition(':')
            if key in names:
                values[key] = int(value.split()[0])
    except (OSError, ValueError):
        pass
    return values


def usage(pids):
    """(bytes, threads): anonymous + shmem + swap memory and thread count over ``pids``."""
    memory = threads = 0
    for pid in pids:
        status = _fields(f'/proc/{pid}/status', ('RssAnon', 'RssShmem', 'VmSwap', 'Threads'))
        memory += 1024 * (status.get('RssAnon', 0) + status.get('RssShmem', 0) + status.get('VmSwap', 0))
        threads += status.get('Threads', 0)
    return memory, threads


def proportional(pids):
    """PSS of the same pages: copy-on-write pages shared by forked processes count once."""
    total = 0
    for pid in pids:
        rollup = _fields(f'/proc/{pid}/smaps_rollup', ('Pss_Anon', 'Pss_Shmem', 'SwapPss'))
        total += 1024 * sum(rollup.values())
    return total


def repin(pids, cpus):
    """Pin back any thread whose affinity left the unit's cores."""
    for pid in pids:
        try:
            tids = [int(t) for t in os.listdir(f'/proc/{pid}/task')]
        except OSError:
            continue
        for tid in tids:
            try:
                if not os.sched_getaffinity(tid) <= cpus:
                    os.sched_setaffinity(tid, cpus)
            except OSError:
                pass


def reap(main, status):
    """Collect every exited child (we are the subreaper); record the main command's exit code."""
    while True:
        try:
            pid, code = os.waitpid(-1, os.WNOHANG)
        except ChildProcessError:
            return
        if pid == 0:
            return
        if pid == main:
            code = os.waitstatus_to_exitcode(code)
            status['code'] = 128 - code if code < 0 else code


def stop(main, status, graceful):
    """Stop the whole tree: TERM, then KILL after STOP_SECONDS (graceful), else KILL at once."""
    me = os.getpid()
    kill_at = time.monotonic() + (STOP_SECONDS if graceful else 0)
    termed = False
    while True:
        reap(main, status)
        pids = [p for p in tree([me]) if p != me]
        if not pids:
            return
        if time.monotonic() >= kill_at:
            sent = signal.SIGKILL
        elif not termed:
            sent, termed = signal.SIGTERM, True
        else:
            sent = None
        for pid in pids if sent else ():
            try:
                os.kill(pid, sent)
            except ProcessLookupError:
                pass
        time.sleep(.02)


def supervise(argv, *, cpus, memory_bytes, seconds, tasks_max, private_tmp=False):
    libc = ctypes.CDLL(None, use_errno=True)
    owner = os.getppid()
    libc.prctl(PR_SET_CHILD_SUBREAPER, 1, 0, 0, 0)
    stopping = []
    signal.signal(signal.SIGTERM, lambda *_: stopping.append('stopped'))
    signal.signal(signal.SIGINT, lambda *_: stopping.append('stopped'))
    if os.getppid() != owner or owner == 1:
        return 143
    cpus = set(cpus)
    os.sched_setaffinity(0, cpus)
    if private_tmp:
        argv = ['bwrap', '--dev-bind', '/', '/', '--tmpfs', '/tmp', '--tmpfs', '/var/tmp',
                '--die-with-parent', '--', *argv]
    env = dict(os.environ, **{LIMIT: str(memory_bytes)})
    main = subprocess.Popen(argv, env=env).pid  # reaped by reap(); never polled through Popen
    deadline = time.monotonic() + seconds
    status, reason, peak = {}, None, 0
    while True:
        reap(main, status)
        if 'code' in status:
            break
        if stopping or os.getppid() != owner:
            reason = 'stopped' if stopping else 'owner died'
            break
        if time.monotonic() >= deadline:
            reason = f'runtime limit {seconds}s'
            break
        pids = [p for p in tree([os.getpid()]) if p != os.getpid()]
        memory, threads = usage(pids)
        if memory > memory_bytes:
            memory = proportional(pids)
        peak = max(peak, memory)
        if memory > memory_bytes:
            reason = f'memory limit {memory_bytes} bytes exceeded ({memory})'
            break
        if threads > tasks_max:
            reason = f'tasks limit {tasks_max} exceeded ({threads})'
            break
        repin(pids, cpus)
        time.sleep(SAMPLE_SECONDS)
    # Like an OOM kill, a broken memory/task limit gets no grace; every other stop does.
    killed = bool(reason) and reason.startswith(('memory', 'tasks'))
    stop(main, status, graceful=not killed)
    code = status.get('code', 137 if killed else 143)
    if reason:
        code = code or (137 if killed else 143)
    print(json.dumps(dict(event='unit_finished', reason=reason or 'exited', exit_code=code,
                          peak_memory_bytes=peak)), file=sys.stderr, flush=True)
    return code


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpus', required=True)
    parser.add_argument('--memory-gib', type=int, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--tasks-max', type=int, required=True)
    parser.add_argument('--private-tmp', action='store_true')
    parser.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    argv = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if not argv:
        parser.error('a command is required')
    raise SystemExit(supervise(argv, cpus=[int(c) for c in args.cpus.split(',')],
                               memory_bytes=args.memory_gib * 1024 ** 3, seconds=args.seconds,
                               tasks_max=args.tasks_max, private_tmp=args.private_tmp))


if __name__ == '__main__':
    main()
