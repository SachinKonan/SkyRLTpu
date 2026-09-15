"""CPU science grading on the existing Ray v2 workload cluster.

Ray reserves logical resources. A distinct systemd cgroup enforces each task's
CPU set, aggregate memory, process count, and lifetime. All prepared nodes,
including TPU inference nodes, are eligible. Candidates never run in Ray.
"""
import fcntl
import json
import os
from pathlib import Path
import pwd
import subprocess
import time
import uuid

import ray


def acquire_slot(root, slots=2, deadline_seconds=2400):
    deadline=time.monotonic()+deadline_seconds
    while time.monotonic()<deadline:
        for slot in range(slots):
            handle=(root/f'cpu-slot-{slot}.lock').open('a')
            try:fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:handle.close();continue
            return slot,handle
        time.sleep(.2)
    raise TimeoutError('CPU admission timed out before candidate execution')


@ray.remote(num_cpus=4,memory=8*1024**3,max_retries=0)
def grade(task, source, root, *, admission_timeout_s=2400):
    from .worker import process_identity
    from .rewards import invalid
    if task not in ('portfolio','portfolio_v2','routing'):raise ValueError('unsupported science task')
    root=Path(root).resolve();jobs=root/'.science/ray-jobs';jobs.mkdir(exist_ok=True)
    if not (root/'.science/ready.json').is_file():raise RuntimeError('CPU worker dependencies not prepared')
    # Admission belongs to the batch queue allowance, not candidate runtime.
    slot,lock=acquire_slot(jobs, deadline_seconds=admission_timeout_s)
    job_id=uuid.uuid4().hex;unit='science-grade-'+job_id
    folder=jobs/job_id;folder.mkdir()
    (folder/'candidate.py').write_text(source)
    (folder/'request.json').write_text(json.dumps(dict(task=task,source=str(folder/'candidate.py'),
        work=str(folder/'evaluation'),root=str(root))))
    seconds=1800 if task=='routing' else 300
    # Reserve a small fixed CPU set for grading; never occupy all host cores.
    cpus=list(range(16+slot*4,20+slot*4))
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
        lock.close()
    result['metrics'].update(ray_node_id=ray.get_runtime_context().get_node_id(),
        host=__import__('socket').gethostname(),job_id=job_id,task=task,ray_executor=True,
        task_envelope_seconds=time.monotonic()-started,hard_memory_gib=8,hard_cpus=cpus,
        artifact_directory=str(folder))
    (folder/'verdict.json').write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')
    return result
