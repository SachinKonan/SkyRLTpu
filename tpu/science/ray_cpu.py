"""CPU science grading on the existing Ray v2 workload cluster.

Ray reserves logical resources. A distinct systemd cgroup enforces each task's
CPU set, aggregate memory, process count, and lifetime. All prepared nodes,
including TPU inference nodes, are eligible. Candidates never run in Ray.
"""
import json
import os
from pathlib import Path
import pwd
import shutil
import subprocess
import time
import uuid

import ray
from .cpu_slots import acquire_slot, slot_cpus, validate_slots


def cleanup_builds(folder):
    """Discard reproducible private builds, retaining source, logs and outputs."""
    work = Path(folder) / 'evaluation'
    if work.is_symlink():
        raise RuntimeError('unexpected evaluation symlink')
    removed = []
    for name in ('target', 'rust'):
        path = work / name
        if path.is_symlink():
            path.unlink()
            removed.append(name)
        elif path.exists():
            shutil.rmtree(path)
            removed.append(name)
    return removed


@ray.remote(num_cpus=4,memory=8*1024**3,max_retries=0)
def grade(task, source, root, *, admission_timeout_s=2400, slots_per_host=2):
    if task not in ('portfolio','portfolio_v2','routing'):raise ValueError('unsupported science task')
    validate_slots(slots_per_host)
    if task != 'routing' and slots_per_host != 2:raise ValueError('expanded slots apply only to routing')
    root=Path(root).resolve();jobs=root/'.science/ray-jobs';jobs.mkdir(exist_ok=True)
    if not (root/'.science/ready.json').is_file():raise RuntimeError('CPU worker dependencies not prepared')
    # Admission belongs to the batch queue allowance, not candidate runtime.
    # Locks are shared across payload directories; never unlink them on release.
    slot,lock=acquire_slot(slots=slots_per_host, deadline_seconds=admission_timeout_s)
    with lock:
        return _grade_admitted(task, source, root, jobs, slot, slots_per_host)


def _grade_admitted(task, source, root, jobs, slot, slots_per_host):
    from .worker import process_identity
    from .rewards import invalid
    job_id=uuid.uuid4().hex;unit='science-grade-'+job_id
    folder=jobs/job_id;folder.mkdir()
    (folder/'candidate.py').write_text(source)
    (folder/'request.json').write_text(json.dumps(dict(task=task,source=str(folder/'candidate.py'),
        work=str(folder/'evaluation'),root=str(root))))
    seconds=1800 if task=='routing' else 300
    # Sixteen disjoint four-CPU sets, leaving CPUs 0-15 and 80+ for the host.
    cpus=slot_cpus(slot)
    if not set(cpus)<=os.sched_getaffinity(0):raise RuntimeError('configured grading CPU set unavailable')
    user=pwd.getpwuid(os.getuid()).pw_name
    command=['sudo','-n','systemd-run','--unit='+unit,'--uid='+user,'--gid='+str(os.getgid()),
        '--wait','--collect','--pipe','--quiet','--property=MemoryMax=8G','--property=MemorySwapMax=0',
        '--property=CPUQuota=400%','--property=AllowedCPUs='+','.join(map(str,cpus)),
        '--property=TasksMax=128','--property=RuntimeMaxSec='+str(seconds),
        '--property=KillMode=control-group','--property=TimeoutStopSec=2','--property=OOMPolicy=stop',
        '--working-directory='+str(root),str(root/'.science/venv/bin/python'),'-m','tpu.science.worker',
        '--request',str(folder/'request.json'),'--result',str(folder/'result.json'),
        '--owner-pid',str(os.getpid()),'--owner-start',process_identity(os.getpid())]
    started=time.monotonic()
    try:
        with (folder/'worker.log').open('wb') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:code=proc.wait(timeout=seconds+20)
            finally:
                if proc.poll() is None:proc.terminate();proc.wait(timeout=5)
        if code:
            result=invalid(f'CPU task exited {code} (timeout, resource limit, or worker failure)',phase='worker')
            result['stdout']=(folder/'worker.log').read_text(errors='replace')[-4000:]
        else:result=json.loads((folder/'result.json').read_text())
    finally:
        # This also executes on cooperative Ray cancellation. A dead Ray worker
        # is detected by the unit's watchdog; RuntimeMaxSec is the final backstop.
        subprocess.run(['sudo','-n','systemctl','stop',unit],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
    if task == 'routing':
        try:
            result['metrics']['removed_build_dirs'] = cleanup_builds(folder)
        except OSError as exc:
            result['metrics']['build_cleanup_error'] = str(exc)
    result['metrics'].update(ray_node_id=ray.get_runtime_context().get_node_id(),
        host=__import__('socket').gethostname(),job_id=job_id,task=task,ray_executor=True,
        task_envelope_seconds=time.monotonic()-started,hard_memory_gib=8,hard_cpus=cpus,
        artifact_directory=str(folder),grading_slots_per_host=slots_per_host,
        grading_memory_cap_gib=8*slots_per_host)
    (folder/'verdict.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')
    return result
