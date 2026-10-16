"""Process lifecycle: one systemd unit per job, and stopping what dead jobs left behind.

TPU hosts. A job runs its workload in one system unit (KillMode=control-group), named from the
SkyPilot task ID and described by it, so stopping the unit kills every process it started: Ray,
engines, ingress and graders. Pool workers run one job at a time, so at startup anything another
task left (its units, workload processes outside the current job's unit, a private executor Ray
cluster, TPU device owners) is a leftover: `prepare` stops it instead of refusing the launch. On a shared
private host nothing is stopped and the launch refuses (see host_claim.py). `finish` is the one end path for
normal exit, error and SkyPilot cancel; exit hooks release leases and cancel in-flight work on the same exits.

Della. Runners, the scheduler and discovery run as systemd user units (`user_unit_argv`), so a
login-session kill cannot take them down and a restart is a unit restart, not setsid + a pid file.
"""
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import time

# Controller exit code for a host stopping ahead of its known end (Slurm time limit). Bootstrap then stays alive
# and idle, so the job ends with its allocation, like a preempted spot VM, and managed-job recovery starts a
# replacement farm. Exiting would end the job as succeeded (no recovery) or failed (an error restart).
HOST_ENDING_EXIT = 75


def idle_until_killed(log, sleep=time.sleep):
    """Nothing runs any more (Ray, Serve and grading are stopped, cores released, /health is gone); wait for the
    allocation to end. SIGTERM is ignored: only the end of the allocation (SIGKILL) may end the job."""
    from .events import emit
    emit(log, 'host_ending_idle')
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, signal.SIG_IGN)
    while True:
        sleep(3600)


UNIT_PATTERNS = ('skyrl-job-*', 'skyrl-runtime-*', 'placement-grade-*', 'science-grade-*', 'math-grade-*')
WORKLOAD_MODULES = {'skyrl.tinker.api', 'skyrl.tinker.engine', 'skyrl.backends.rpc',
                    'tpu.thinking_budget.server', 'tpu.swarm.ray_train.thinking_budget.server',
                    'tpu.swarm.ray_train.grader_child', 'tpu.swarm.ray_train.bootstrap',
                    'tpu.science.placement_task', 'tpu.science.challenge_score_child', 'tpu.science.worker'}
WORKLOAD_SUFFIXES = ('/vllm_tpu_server.py', '/thinking_budget/server.py', '/grader_child.py', '/runner.py')


def job_unit(task_id):
    if not task_id:
        raise ValueError('a job unit needs the SkyPilot task ID')
    return 'skyrl-job-' + hashlib.sha256(task_id.encode()).hexdigest()[:20] + '.service'


def job_unit_argv(task_id, argv, log, cwd, stop_grace=60):
    """Run argv as the job's unit and wait for it; the unit, not a PID, owns the process tree."""
    return ['sudo', '-n', 'systemd-run', '--quiet', '--wait', '--collect', '--unit=' + job_unit(task_id),
            '--description=' + task_id, '--uid=' + str(os.getuid()), '--gid=' + str(os.getgid()),
            '--property=Type=exec', '--property=KillMode=control-group', '--property=SendSIGKILL=yes',
            '--property=TimeoutStopSec=' + str(stop_grace), '--property=TasksMax=infinity',
            '--property=LimitNOFILE=1048576', '--property=LimitMEMLOCK=infinity',
            '--property=StandardOutput=append:' + str(log), '--property=StandardError=append:' + str(log),
            '--working-directory=' + str(cwd), *argv]


def user_unit_argv(name, argv, cwd, log, env=None, restart='no', properties=()):
    """A della-side long-running process as a systemd user unit (linger keeps it past logout).

    `properties` adds unit settings (e.g. Type=notify + WatchdogSec for a service systemd should restart
    when it stops reporting); a Type= there replaces the default Type=exec.
    """
    if not name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('invalid unit name')
    setenv = ['--setenv=%s=%s' % kv for kv in sorted((env or {}).items())]
    kind = [] if any(p.startswith('Type=') for p in properties) else ['--property=Type=exec']
    return ['systemd-run', '--user', '--quiet', '--collect', '--unit=' + name, *kind,
            '--property=KillMode=control-group', '--property=Restart=' + restart,
            *['--property=' + p for p in properties],
            '--property=StandardOutput=append:' + str(log), '--property=StandardError=append:' + str(log),
            '--working-directory=' + str(cwd), *setenv, *argv]


# In-process exit hooks: the client/ingress register lease release and in-flight cancel here, and
# install_exit_handlers() runs them once on every exit a process can observe: normal return (atexit), an
# unhandled error, and SIGTERM/SIGINT, which is what both `systemctl stop` (KillMode=control-group signals
# every process in the unit) and a SkyPilot cancel deliver. Preemption kills the VM with no signal: farms
# expire the lease by TTL and the next attempt's prepare() clears the host.
_EXIT_HOOKS = []
_EXIT_STATE = dict(ran=False, log=None, installed=False)


def on_exit(name, hook):
    """Register hook(reason) to run once at process exit (release a lease, cancel in-flight requests)."""
    _EXIT_HOOKS.append((name, hook))
    return hook


def run_exit_hooks(reason):
    """Run every registered hook once, logging each; a failing hook never stops the others."""
    if _EXIT_STATE['ran']:
        return []
    _EXIT_STATE['ran'] = True
    results = []
    for name, hook in list(_EXIT_HOOKS):
        try:
            hook(reason)
            results.append(dict(hook=name, ok=True))
        except Exception as exc:
            results.append(dict(hook=name, ok=False, error=type(exc).__name__ + ': ' + str(exc)[:200]))
    if _EXIT_STATE['log'] is not None and results:
        from .events import emit
        for item in results:
            emit(_EXIT_STATE['log'], 'exit_hook', reason=reason, **item)
    return results


def install_exit_handlers(log=None):
    """Run the exit hooks on normal exit, unhandled errors and SIGTERM/SIGINT, then the previous handler."""
    import atexit
    import signal
    import sys
    _EXIT_STATE['log'] = log
    if _EXIT_STATE['installed']:
        return
    _EXIT_STATE['installed'] = True
    atexit.register(run_exit_hooks, 'exit')
    previous_hook = sys.excepthook

    def excepthook(kind, value, tb):
        run_exit_hooks('error')
        previous_hook(kind, value, tb)
    sys.excepthook = excepthook
    for sig in (signal.SIGTERM, signal.SIGINT):
        previous = signal.getsignal(sig)

        def handler(signum, frame, previous=previous):
            run_exit_hooks('signal')
            if callable(previous):
                previous(signum, frame)
            elif previous == signal.SIG_DFL:
                signal.signal(signum, signal.SIG_DFL)
                os.kill(os.getpid(), signum)
        signal.signal(sig, handler)


def active_units(run=subprocess.run):
    """[(unit, description)] of our unit families that are still active on this host."""
    out = run(['systemctl', 'list-units', '--no-legend', '--plain', '--full',
               '--state=active,activating,deactivating', *UNIT_PATTERNS],
              capture_output=True, text=True, timeout=15)
    if out.returncode:
        raise RuntimeError('unit audit failed: ' + out.stderr.strip()[-300:])
    units = []
    for line in out.stdout.splitlines():
        fields = line.split(None, 4)
        if fields:
            units.append((fields[0], fields[4] if len(fields) > 4 else ''))
    return units


def workload_processes(proc=Path('/proc'), me=None):
    """[(pid, cgroup, reason)] of workload processes: model servers, trainers, graders, executor Ray."""
    me = os.getpid() if me is None else me
    found = []
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == me:
            continue
        try:
            args = entry.joinpath('cmdline').read_bytes().decode(errors='replace').strip('\0').split('\0')
            cgroup = entry.joinpath('cgroup').read_text().strip().rsplit(':', 1)[-1]
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        reason = next((a for a in args if a in WORKLOAD_MODULES or a.endswith(WORKLOAD_SUFFIXES)
                       or a.startswith('VLLM::')), None)
        # A cancelled job can leave its private executor Ray head; SkyPilot's own Ray (ray_skypilot) is not ours.
        if reason is None and args and Path(args[0]).name in ('raylet', 'gcs_server') and any(
                '/.rtr-' in a for a in args):
            reason = 'private executor Ray cluster'
        if reason:
            found.append((int(entry.name), cgroup, reason))
    return found


def prepare(task_id, *, shared=False, log=None, tpu=True, run=subprocess.run, proc=Path("/proc"), kill=os.kill,
            scope=None):
    """The one start path: a fresh launch and a preemption recovery both begin here.

    Pool workers (shared=False) stop whatever earlier tasks left. On a shared private host nothing is stopped:
    any leftover makes the launch refuse, and host_claim.py decides ownership.
    """
    report = reap_leftovers(task_id, stop=not shared, run=run, proc=proc, kill=kill, scope=scope)
    if tpu:
        report.update(tpu_gates(shared=shared, run=run))
    if log is not None:
        from .events import emit
        emit(log, 'lifecycle_prepared', task_id=task_id, shared=shared, **report)
    if shared and (report['units'] or report['processes'] or report.get('device_owners')
                   or report.get('foreign_tpu_lock')):
        raise RuntimeError('shared host has workloads from another task; refusing to start: '
                           + json.dumps(report))
    return report


def tpu_gates(*, shared, run=subprocess.run, home=None, tmp=Path('/tmp'), dev=Path('/dev')):
    """What clean_host_audit checked, healed where a pool worker allows it.

    - TPU device owners: on a pool worker they belong to a dead task and are killed (sudo fuser -k); a shared
      host reports them (prepare refuses).
    - /tmp/libtpu_lockfile left by another login user (libtpu then says "already in use"): removed on a pool
      worker once no process owns a device; reported on a shared host.
    - Space: home >= 30 GiB and /tmp >= 10 GiB free with >= 10000 inodes, else refuse. Host preflight's storage
      reconcile normally keeps home above its target, so this only fires when reclamation cannot help.
    """
    import shutil
    home = Path.home() if home is None else home
    for name, path, minimum in (('home', home, 30), ('tmp', tmp, 10)):
        usage, inodes = shutil.disk_usage(path), os.statvfs(path).f_favail
        if usage.free < minimum * 1024**3 or inodes < 10000:
            raise RuntimeError(f'insufficient storage on {name}: {usage.free / 1024**3:.1f} GiB free, '
                               f'{inodes} inodes')
    devices = [str(p) for p in (dev / 'vfio').glob('*') if p.name != 'vfio'] + [str(p) for p in dev.glob('accel*')]
    if not devices:
        raise RuntimeError('no TPU devices found')
    owners = run(['sudo', '-n', 'fuser', *devices], capture_output=True, text=True, timeout=30)
    held = owners.returncode == 0 or bool((owners.stdout or '').strip())
    if owners.returncode not in (0, 1):
        raise RuntimeError('TPU device audit failed: ' + (owners.stderr or '').strip()[-300:])
    if held and not shared:
        run(['sudo', '-n', 'fuser', '-k', '-KILL', *devices], capture_output=True, text=True, timeout=30)
    lock = tmp / 'libtpu_lockfile'
    foreign = lock.exists() and not os.access(lock, os.R_OK | os.W_OK)
    if foreign and not shared:
        run(['sudo', '-n', 'rm', '-f', str(lock)], capture_output=True, text=True, timeout=15)
    return dict(device_owners=(owners.stdout or '').split() if held else [], foreign_tpu_lock=foreign)


def finish(task_id, reason, *, shared=False, log=None, run=subprocess.run, proc=Path('/proc'), kill=os.kill):
    """The one end path for normal exit, error exit and SkyPilot cancel (SIGTERM).

    Called after the job's runtime unit has stopped, so systemd has already torn down its cgroup. This stops any
    unit of ours still tagged with this task (graders bound to a dead runtime, a unit stuck deactivating) and,
    on pool workers, any workload process that escaped every unit. Preemption has no end path (the VM is gone);
    the next attempt's prepare() covers it.
    """
    run_exit_hooks(reason)  # this process's own lease/in-flight hooks, if it registered any
    units = [u for u, d in active_units(run) if d == task_id or (task_id and u == job_unit(task_id))]
    for unit in units:
        run(['sudo', '-n', 'systemctl', 'stop', unit], capture_output=True, text=True, timeout=180)
    strays = []
    if not shared:
        live = {u for u, _ in active_units(run)}
        strays = [(pid, why) for pid, cg, why in workload_processes(proc)
                  if not any(cg.endswith('/' + u) for u in live)]
        for pid, _ in strays:
            try:
                kill(pid, 9)
            except ProcessLookupError:
                pass
            except PermissionError:
                run(['sudo', '-n', 'kill', '-9', str(pid)], capture_output=True, text=True, timeout=15)
    report = dict(reason=reason, units=units, processes=[dict(pid=p, reason=r) for p, r in strays])
    if log is not None:
        from .events import emit
        emit(log, 'lifecycle_finished', task_id=task_id, **report)
    return report


def reap_leftovers(task_id, *, stop=True, run=subprocess.run, proc=Path('/proc'), kill=os.kill, scope=None):
    """Find (and by default stop) what other tasks left on this host. Returns what was found.

    Call before the job starts any workload: every active unit of ours that is not the current job's,
    and every workload process outside the current job's unit, belongs to a task that is gone.
    Processes inside a leftover unit die with `systemctl stop`; only those outside every unit
    (escaped, or started before units existed) are killed directly.
    """
    current = job_unit(task_id) if task_id else None
    units = [u for u, d in active_units(run) if u != current and d != task_id]
    owned = ([current] if current else []) + units
    # scope (a Slurm job id): on a shared Slurm node only our own allocation's processes are ours to reap.
    strays = [(pid, cg, why) for pid, cg, why in workload_processes(proc)
              if not any(cg.endswith('/' + u) for u in owned) and (not scope or f'/job_{scope}/' in cg + '/')]
    report = dict(units=units, processes=[dict(pid=p, cgroup=c, reason=r) for p, c, r in strays])
    if not stop:
        return report
    for unit in units:
        run(['sudo', '-n', 'systemctl', 'stop', unit], capture_output=True, text=True, timeout=180)
    for pid, _, _ in strays:
        try:
            kill(pid, 9)
        except ProcessLookupError:
            pass
        except PermissionError:
            run(['sudo', '-n', 'kill', '-9', str(pid)], capture_output=True, text=True, timeout=15)
    return report


PASSED_ENV_PREFIXES = ('TTD_', 'SKYRL_', 'SKYPILOT_', 'PYTHON', 'OPENBLAS_', 'OMP_', 'MKL_', 'HF_', 'UV_')
PASSED_ENV_KEYS = ('HOME', 'PATH', 'USER', 'LANG', 'NUM_EPOCHS', 'FARM_GROUP_CAP', 'FARM_TARGET')


def start_user_unit(name, argv, cwd, log, *, environ=None, restart='no', run=subprocess.run):
    """Start a della-side process as a user unit; returns its main PID (for existing pid-file checks).

    A user unit does not inherit the caller's shell environment, so the variables runs depend on (TTD_*,
    SKYRL_*, PYTHONPATH, thread caps, ...) are passed explicitly.
    """
    environ = os.environ if environ is None else environ
    env = {k: v for k, v in environ.items() if k in PASSED_ENV_KEYS or k.startswith(PASSED_ENV_PREFIXES)}
    run(['systemctl', '--user', 'reset-failed', name], capture_output=True, text=True, timeout=15)
    out = run(user_unit_argv(name, argv, cwd, log, env=env, restart=restart), capture_output=True, text=True,
              timeout=60)
    if out.returncode:
        raise RuntimeError('user unit failed to start: ' + (out.stderr or '').strip()[-300:])
    show = run(['systemctl', '--user', 'show', name, '-p', 'MainPID', '--value'], capture_output=True, text=True,
               timeout=15)
    return int((show.stdout or '0').strip() or 0)


def main(argv=None):
    import argparse
    import sys
    parser = argparse.ArgumentParser(description='Job lifecycle: audit leftovers, start della user units.')
    sub = parser.add_subparsers(dest='command', required=True)
    audit = sub.add_parser('audit', help='report (never stop) what other tasks left on this host')
    audit.add_argument('--task-id', default=os.environ.get('SKYPILOT_TASK_ID', ''))
    prep = sub.add_parser('prepare', help='the start path for launchers that do not go through bootstrap')
    prep.add_argument('--task-id', default=os.environ.get('SKYPILOT_TASK_ID', ''))
    prep.add_argument('--shared', action='store_true')
    unit = sub.add_parser('user-unit', help='run a command as a systemd user unit and print its main PID')
    unit.add_argument('--name', required=True)
    unit.add_argument('--cwd', required=True)
    unit.add_argument('--log', required=True)
    unit.add_argument('--restart', default='no')
    unit.add_argument('argv', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.command == 'audit':
        print(json.dumps(reap_leftovers(args.task_id, stop=False), indent=1))
        return 0
    if args.command == 'prepare':
        print(json.dumps(dict(event='lifecycle_prepared', **prepare(args.task_id, shared=args.shared))), flush=True)
        return 0
    command = args.argv[1:] if args.argv[:1] == ['--'] else args.argv
    if not command:
        parser.error('user-unit needs a command after --')
    print(start_user_unit(args.name, command, args.cwd, args.log, restart=args.restart))
    sys.stdout.flush()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
