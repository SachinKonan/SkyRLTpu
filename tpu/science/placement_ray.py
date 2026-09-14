"""Placement execution on private Ray v2 clusters, outside serving TPU ranks."""
import json
import os
from pathlib import Path
import pwd
import subprocess
import time
import uuid

import ray
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy
from ray.util.placement_group import placement_group, remove_placement_group
from .placement_slots import chip_lock, chip_cpus, task_resources, assigned_chip, grading_nodes, grading_bundles


@ray.remote(num_cpus=4, memory=16*1024**3, resources=task_resources(), max_retries=0)
def grade_case(source, case, root, backend='tpu', *, accelerator='tpu-v4-64'):
    chip = assigned_chip(ray.get_runtime_context().get_accelerator_ids())
    with chip_lock(chip):
        return _grade_case(source, case, root, backend, chip, accelerator)


class PlacementPool:
    """One packed placement group per grading host; Ray assigns task chip IDs."""
    def __init__(self, nodes, *, timeout=120):
        self.groups = []
        self.pending = []
        self.next_group = 0
        try:
            for node in grading_nodes(nodes):
                self.groups.append(placement_group(grading_bundles(node), strategy='STRICT_PACK'))
            if not self.groups:
                raise RuntimeError('no grading hosts registered')
            ray.get([group.ready() for group in self.groups], timeout=timeout)
        except BaseException:
            self.close()
            raise

    def submit(self, source, case, root, *, accelerator='tpu-v4-64'):
        group = self.groups[self.next_group % len(self.groups)]
        self.next_group += 1
        ref = grade_case.options(scheduling_strategy=PlacementGroupSchedulingStrategy(
            placement_group=group, placement_group_bundle_index=-1,
            placement_group_capture_child_tasks=False)).remote(source, case, root, accelerator=accelerator)
        self.pending.append(ref)
        return ref

    def close(self):
        try:
            for ref in self.pending:
                ray.cancel(ref, force=True)
        finally:
            for group in self.groups:
                remove_placement_group(group)
            self.groups.clear()
            self.pending.clear()


def _grade_case(source, case, root, backend, chip, accelerator):
    from .worker import process_identity
    from .rewards import invalid
    root=Path(root).resolve(); jobs=root/'.science/placement-jobs';jobs.mkdir(exist_ok=True)
    folder=jobs/uuid.uuid4().hex;folder.mkdir();unit='placement-grade-'+folder.name
    (folder/'candidate.py').write_text(source)
    request=dict(source=str(folder/'candidate.py'),case=case,root=str(root),
                 work=str(folder/'evaluation'),backend=backend,tpu_ids=[str(chip)],accelerator=accelerator)
    (folder/'request.json').write_text(json.dumps(request))
    cpus=chip_cpus(chip)
    if not set(cpus)<=os.sched_getaffinity(0):raise RuntimeError('placement CPU set unavailable')
    user=pwd.getpwuid(os.getuid()).pw_name
    # TPU driver mappings need a permissive memlock limit. MemoryMax still
    # bounds the candidate's actual host RAM across its entire process tree.
    cmd=['sudo','-n','systemd-run','--unit='+unit,'--uid='+user,'--gid='+str(os.getgid()),
         '--wait','--collect','--pipe','--quiet','--property=MemoryMax=16G','--property=MemorySwapMax=0',
         '--property=LimitMEMLOCK=infinity',
         '--property=CPUQuota=400%','--property=AllowedCPUs='+','.join(map(str,cpus)),
         '--property=TasksMax=1024','--property=RuntimeMaxSec=300','--property=KillMode=control-group',
         '--property=TimeoutStopSec=2','--property=OOMPolicy=stop','--working-directory='+str(root),
         str(root/'.science/venv/bin/python'),'-m','tpu.science.placement_task',
         '--request',str(folder/'request.json'),'--result',str(folder/'result.json'),
         '--owner-pid',str(os.getpid()),'--owner-start',process_identity(os.getpid())]
    started=time.monotonic()
    try:
        with (folder/'worker.log').open('wb') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            try: code=proc.wait(timeout=320)
            finally:
                if proc.poll() is None: proc.terminate();proc.wait(timeout=5)
        if code or not (folder/'result.json').exists():
            result=invalid(f'placement worker exited {code}',phase='worker')
            result['stdout']=(folder/'worker.log').read_text(errors='replace')[-4000:]
        else:result=json.loads((folder/'result.json').read_text())
    finally:
        subprocess.run(['sudo','-n','systemctl','stop',unit],stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL,timeout=10)
    result['metrics'].update(ray_node_id=ray.get_runtime_context().get_node_id(),
        host=__import__('socket').gethostname(),case=case,artifact_directory=str(folder),
        task_envelope_seconds=time.monotonic()-started,hard_memory_gib=16,hard_cpus=cpus,
        physical_chip_id=chip,ray_tpu_ids=[str(chip)],accelerator=accelerator,physical_tpu_chips=1 if backend=='tpu' else 0,ray_executor=True)
    (folder/'verdict.json').write_text(json.dumps(result,allow_nan=False,indent=2))
    return result



from .challenge_contract import aggregate
