"""Process lifecycle: one systemd unit per job, and stopping what dead jobs left behind.

TPU hosts. A job runs its workload in one system unit (KillMode=control-group), named from the
SkyPilot task ID and described by it, so stopping the unit kills every process it started: Ray,
engines, ingress and graders. Pool workers run one job at a time, so at startup anything another
task left (its units, workload processes outside the current job's unit, a private executor Ray
cluster) is a leftover: `reap_leftovers` stops it instead of refusing the launch, which today
needs a manual clean_worker.sh. On a shared private host the caller passes `stop=False` and gets
the list back, and the launch refuses (see host_claim.py).

Della. Runners, the scheduler and discovery run as systemd user units (`user_unit_argv`), so a
login-session kill cannot take them down and a restart is a unit restart, not setsid + a pid file.
"""
import hashlib
import os
from pathlib import Path
import subprocess

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


def user_unit_argv(name, argv, cwd, log, env=None, restart='no'):
    """A della-side long-running process as a systemd user unit (linger keeps it past logout)."""
    if not name.replace('-', '').replace('_', '').isalnum():
        raise ValueError('invalid unit name')
    setenv = ['--setenv=%s=%s' % kv for kv in sorted((env or {}).items())]
    return ['systemd-run', '--user', '--quiet', '--collect', '--unit=' + name, '--property=Type=exec',
            '--property=KillMode=control-group', '--property=Restart=' + restart,
            '--property=StandardOutput=append:' + str(log), '--property=StandardError=append:' + str(log),
            '--working-directory=' + str(cwd), *setenv, *argv]


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


def reap_leftovers(task_id, *, stop=True, run=subprocess.run, proc=Path('/proc'), kill=os.kill):
    """Find (and by default stop) what other tasks left on this host. Returns what was found.

    Call before the job starts any workload: every active unit of ours that is not the current job's,
    and every workload process outside the current job's unit, belongs to a task that is gone.
    Processes inside a leftover unit die with `systemctl stop`; only those outside every unit
    (escaped, or started before units existed) are killed directly.
    """
    current = job_unit(task_id) if task_id else None
    units = [u for u, d in active_units(run) if u != current and d != task_id]
    owned = ([current] if current else []) + units
    strays = [(pid, cg, why) for pid, cg, why in workload_processes(proc)
              if not any(cg.endswith('/' + u) for u in owned)]
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
