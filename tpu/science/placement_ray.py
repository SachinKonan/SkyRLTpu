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
from .cpu_slots import acquire_slot, slot_cpus
from .challenge_contract import TASK_ENVELOPE_SECONDS


@ray.remote(num_cpus=4, memory=8*1024**3, resources={'placement_cpu_host': 1}, max_retries=0)
def grade_cpu_case(source, case, root, *, slots_per_host=16, admission_timeout_s=2400, helper='none', resource_contract=None):
    """One case on host CPUs; no Ray TPU request and no device mounts."""
    if resource_contract is not None:
        from .placement_resources import validate, acquire
        validate(resource_contract)
        if resource_contract['slots'] != slots_per_host:
            raise ValueError('placement admission differs from resource contract')
        slot, cpus, lock = acquire(slots_per_host, deadline_seconds=admission_timeout_s)
        with lock:
            return _grade_case(source, case, root, 'cpu-jax', None, None,
                cpu_slot=slot, slots_per_host=slots_per_host, helper=helper,
                resource_contract=resource_contract, reserved_cpus=cpus)
    slot, lock = acquire_slot(slots=slots_per_host, deadline_seconds=admission_timeout_s)
    with lock:
        return _grade_case(source, case, root, 'cpu-jax', None, None,
                           cpu_slot=slot, slots_per_host=slots_per_host, helper=helper)


@ray.remote(num_cpus=4, memory=16*1024**3, resources=task_resources(), max_retries=0)
def grade_case(source, case, root, backend='tpu', *, accelerator='tpu-v4-64'):
    chip = assigned_chip(ray.get_runtime_context().get_accelerator_ids())
    visible = assigned_chip({'TPU': os.environ.get('TPU_VISIBLE_CHIPS', '').split(',')})
    if visible != chip:
        raise RuntimeError('Ray TPU visibility does not match its physical allocation')
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


def _grade_case(source, case, root, backend, chip, accelerator, *, cpu_slot=None, slots_per_host=None, helper='none', resource_contract=None, reserved_cpus=None):
    if helper not in ('none', 'fast_proxy_v1') or (helper != 'none' and backend != 'cpu-jax'):
        raise ValueError('fast proxy requires CPU placement')
    from .worker import process_identity
    from .rewards import invalid
    root=Path(root).resolve(); jobs=root/'.science/placement-jobs';jobs.mkdir(exist_ok=True)
    folder=jobs/uuid.uuid4().hex;folder.mkdir();unit='placement-grade-'+folder.name
    (folder/'candidate.py').write_text(source)
    cpu = backend == 'cpu-jax'
    memory_gib = resource_contract["memory_gib"] if resource_contract else (8 if cpu else 16)
    envelope_seconds = resource_contract["envelope_seconds"] if resource_contract else TASK_ENVELOPE_SECONDS
    request=dict(source=str(folder/'candidate.py'),case=case,root=str(root),
                 work=str(folder/'evaluation'),backend=backend,tpu_ids=[] if cpu else [str(chip)],
                 accelerator=accelerator,memory_gib=memory_gib,helper=helper,resource_contract=resource_contract)
    (folder/'request.json').write_text(json.dumps(request))
    cpus=reserved_cpus if resource_contract else (slot_cpus(cpu_slot) if cpu else chip_cpus(chip))
    if not set(cpus)<=os.sched_getaffinity(0):raise RuntimeError('placement CPU set unavailable')
    user=pwd.getpwuid(os.getuid()).pw_name
    # TPU driver mappings need a permissive memlock limit. MemoryMax still
    # bounds the candidate's actual host RAM across its entire process tree.
    from .cgroup_limits import runtime_owner_properties
    cmd=['sudo','-n','systemd-run','--unit='+unit,'--uid='+user,'--gid='+str(os.getgid()),
         '--wait','--collect','--pipe','--quiet',*runtime_owner_properties(),
         f'--property=MemoryMax={memory_gib}G','--property=MemorySwapMax=0',
         '--property=LimitMEMLOCK=infinity',
         '--property=CPUQuota=400%','--property=AllowedCPUs='+','.join(map(str,cpus)),
         '--property=TasksMax=1024',f'--property=RuntimeMaxSec={envelope_seconds}','--property=KillMode=control-group',
         '--property=TimeoutStopSec=2','--property=OOMPolicy=stop','--working-directory='+str(root),
         str(root/'.science/venv/bin/python'),'-m','tpu.science.placement_task',
         '--request',str(folder/'request.json'),'--result',str(folder/'result.json'),
         '--owner-pid',str(os.getpid()),'--owner-start',process_identity(os.getpid())]
    started=time.monotonic()
    try:
        with (folder/'worker.log').open('wb') as log:
            proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            try: code=proc.wait(timeout=envelope_seconds+20)
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
        task_envelope_seconds=time.monotonic()-started,hard_memory_gib=memory_gib,hard_cpus=cpus,
        physical_chip_id=chip,ray_tpu_ids=[] if cpu else [str(chip)],ray_tpu_visible_chips=os.environ.get('TPU_VISIBLE_CHIPS'),
        accelerator=accelerator,physical_tpu_chips=1 if backend=='tpu' else 0,ray_executor=True)
    if cpu:
        result['metrics'].update(grading_slots_per_host=slots_per_host,
                                 grading_memory_cap_gib=memory_gib*slots_per_host,resource_contract=resource_contract, cpu_slot=cpu_slot)
    (folder/'verdict.json').write_text(json.dumps(result,allow_nan=False,indent=2))
    return result



from .challenge_contract import aggregate
