"""AC2 candidate execution as a Ray task with real resource limits.

Mirrors ``ray_cpu._grade_admitted``: Ray reserves logical resources, a
host-wide flock admits the candidate to one CPU slot, and a transient systemd
unit enforces the CPU set, memory, process count, lifetime and a private
``/tmp``. The candidate never runs inside the Ray worker. Used by trainer
hosts (local pool) and by farm hosts (behind the lease-fenced grading
endpoints); both sides share the same executor and result shape.

Return shape: ``{"result", "error", "stdout", "metrics"}`` where ``error`` is
a candidate failure (timeout, crash, exception, non-finite output) and the
reward stays zero on the trainer. Infrastructure failures (admission timeout,
unit died before the candidate started, missing result) raise
``GradingInfrastructureFailure`` so callers retry elsewhere.
"""
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time
import uuid

import ray

from .farm_resources import acquire_ac2
from . import core_pool

STARTED_MARKER = 'started.json'


class GradingInfrastructureFailure(RuntimeError):
    """The grader could not run the candidate; this is not a candidate grade."""


def default_root():
    return Path(os.environ.get('SCIENCE_WORKER_ROOT') or os.getcwd()).resolve()


def unit_command(unit, root, request_path, result_path, cpus, memory_gib, seconds, owner_pid, owner_start):
    from .cgroup_limits import runtime_owner_properties
    user = pwd.getpwuid(os.getuid()).pw_name
    return ['sudo', '-n', 'systemd-run', '--unit=' + unit, '--uid=' + user, '--gid=' + str(os.getgid()),
            '--wait', '--collect', '--pipe', '--quiet', *runtime_owner_properties(),
            '--property=MemoryMax=' + str(memory_gib) + 'G', '--property=MemorySwapMax=0',
            '--property=CPUQuota=' + str(100 * len(cpus)) + '%', '--property=AllowedCPUs=' + ','.join(map(str, cpus)),
            '--property=TasksMax=256', '--property=RuntimeMaxSec=' + str(seconds),
            '--property=KillMode=control-group', '--property=TimeoutStopSec=2', '--property=OOMPolicy=stop',
            '--property=PrivateTmp=yes', '--working-directory=' + str(root),
            sys.executable, '-m', 'tpu.science.ac2_runner', '--request', str(request_path),
            '--result', str(result_path), '--owner-pid', str(owner_pid), '--owner-start', owner_start]


def admit(families, deadline_seconds):
    """(slot, cpus, lease, kind): the host core pool when installed, else the AC2 partition."""
    family = families['ac2']
    try:
        cpus, lease = core_pool.acquire(family['cpus'], family['memory_gib'], deadline_seconds=deadline_seconds)
        return None, cpus, lease, core_pool.VERSION
    except core_pool.PoolUnavailable:
        slot, cpus, lease = acquire_ac2(families, deadline_seconds=deadline_seconds)
        return slot, cpus, lease, 'farm-ac2-v1'


def grade_ac2_admitted(spec, families, root=None):
    """Run one prepared AC2 candidate under the host partition's limits."""
    from .worker import process_identity
    root = Path(root) if root else default_root()
    families = dict(families)
    family = families['ac2']
    systemd = bool(spec.get('systemd', True))
    eval_timeout = int(spec['eval_timeout_seconds'])
    queued = time.monotonic()
    try:
        slot, cpus, lease, admission = admit(families, int(spec.get('admission_timeout_s') or eval_timeout))
    except TimeoutError as exc:
        raise GradingInfrastructureFailure(f'admission timed out: {exc}') from exc
    waited = time.monotonic() - queued
    with lease:
        job_id = uuid.uuid4().hex
        unit = 'ac2-grade-' + job_id
        folder = root / '.science/ac2-jobs' / job_id
        folder.mkdir(parents=True)
        request = dict(program_code=spec['program_code'], function_name=spec['function_name'], cpus=cpus,
                       eval_timeout_seconds=eval_timeout, stdout_limit_bytes=int(spec.get('stdout_limit_bytes') or 16384),
                       memory_gib=int(family['memory_gib']), systemd=systemd)
        request_path, result_path = folder / 'request.json', folder / 'result.json'
        request_path.write_text(json.dumps(request))
        seconds = eval_timeout + 15
        owner = process_identity(os.getpid())
        if systemd:
            command = unit_command(unit, root, request_path, result_path, cpus, family['memory_gib'], seconds,
                                   os.getpid(), owner)
        else:
            command = [sys.executable, '-m', 'tpu.science.ac2_runner', '--request', str(request_path),
                       '--result', str(result_path), '--owner-pid', str(os.getpid()), '--owner-start', owner]
        started = time.monotonic()
        try:
            with (folder / 'worker.log').open('wb') as log:
                proc = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, cwd=str(root))
                try:
                    code = proc.wait(timeout=seconds + 20)
                finally:
                    if proc.poll() is None:
                        proc.terminate()
                        proc.wait(timeout=5)
        finally:
            if systemd:
                # Also runs on cooperative Ray cancellation; a SIGKILLed worker
                # is caught by the unit's owner watchdog and RuntimeMaxSec.
                subprocess.run(['sudo', '-n', 'systemctl', 'stop', unit], stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=10)
        log_tail = (folder / 'worker.log').read_text(errors='replace')[-4000:]
        if not (folder / STARTED_MARKER).is_file():
            raise GradingInfrastructureFailure(f'AC2 unit exited {code} before the candidate started: {log_tail[-800:]}')
        if result_path.is_file():
            result = json.loads(result_path.read_text())
        elif code:
            result = dict(result=None, error=f'AC2 unit exited {code} (timeout, resource limit, or worker failure)',
                          stdout=log_tail, metrics={})
        else:
            raise GradingInfrastructureFailure('AC2 unit finished without a result file')
        metrics = result.setdefault('metrics', {})
        metrics.update(admission_wait_seconds=waited, task_envelope_seconds=time.monotonic() - started,
                       hard_cpus=cpus, hard_memory_gib=family['memory_gib'], slot=slot, job_id=job_id,
                       unit=unit if systemd else None, host=__import__('socket').gethostname(), admission=admission,
                       grading_slots_per_host=family['slots_per_host'], ray_executor=True)
        if ray.is_initialized():  # get_runtime_context() would start a local Ray otherwise.
            try:
                metrics['ray_node_id'] = ray.get_runtime_context().get_node_id()
            except Exception:
                pass
        (folder / 'verdict.json').write_text(json.dumps(result, allow_nan=False))
        return result


grade_ac2 = ray.remote(num_cpus=2, memory=4 * 1024 ** 3, max_retries=0)(grade_ac2_admitted)
