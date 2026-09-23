"""Managed CPU-only validation of the routing evaluator on an existing slice.

Each SkyPilot rank owns one bounded unit; candidate execution stays in bwrap.
No trainer, adapter, farm lease, or TPU runtime is started by this entrypoint.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import pwd
import subprocess
import sys
import time
import uuid

from .routing_regrade import save


def child(request):
    from .worker import watch_owner
    from .rewards import invalid
    from .routing_parallel import cpu_seconds,partial_metrics
    watch_owner(request['owner_pid'],request['owner_start'])
    root=Path(request['root']);work=Path(request['work']);mode=request['mode']
    from .cgroup_limits import envelope,metrics
    group,_=envelope(8 if mode=='legacy' else 20)
    if mode=='legacy':from .routing import evaluate
    else:from .routing_parallel import evaluate
    start=time.monotonic()
    try:
        verdict=evaluate(request['source'],root=root/'.science/routing-task',work=work,
            python=sys.executable,cargo_home=root/'.science/cargo',rustup_home=root/'.science/rustup',
            target_cache=root/'.science/router-target',seconds=request['seconds'],
            **({} if mode=='legacy' else dict(case_workers=1 if mode=='single' else 4)))
    except Exception as exc:
        verdict=invalid(type(exc).__name__+': '+str(exc))
        if mode!='legacy':verdict['metrics'].update(partial_metrics(work,
            deadline_exceeded='shared routing evaluation deadline' in str(exc)))
    verdict['metrics'].update(process_cpu_seconds=cpu_seconds(),benchmark_wall_seconds=time.monotonic()-start,
        allocation_memory=metrics(group),source_sha256=hashlib.sha256(request['source'].encode()).hexdigest())
    from .feedback import observation
    verdict['feedback']=observation('routing',verdict)
    save(request['result'],verdict)


def evaluate_one(root,output,source,mode,seconds,label,slots=1):
    from .routing_resources import acquire
    from .worker import process_identity
    folder=output/label;folder.mkdir(parents=True,exist_ok=False)
    queued=time.monotonic();slot,cpus,lease=acquire(slots=slots,deadline_seconds=2400)
    with lease:
        waited=time.monotonic()-queued;unit='routing-benchmark-'+uuid.uuid4().hex
        request=dict(root=str(root),work=str(folder/'evaluation'),result=str(folder/'result.json'),
            source=source,mode=mode,seconds=seconds,owner_pid=os.getpid(),owner_start=process_identity(os.getpid()))
        save(folder/'request.json',request)
        if mode=='legacy':cpus=cpus[:4]
        cmd=['sudo','-n','systemd-run','--unit='+unit,'--uid='+pwd.getpwuid(os.getuid()).pw_name,
            '--gid='+str(os.getgid()),'--wait','--collect','--pipe','--quiet',
            '--property=MemoryMax='+('8G' if mode=='legacy' else '20G'),'--property=MemorySwapMax=0',
            '--property=Delegate=cpu cpuset memory pids','--property=CPUQuota='+str(100*len(cpus))+'%',
            '--property=AllowedCPUs='+','.join(map(str,cpus)),'--property=TasksMax=640',
            '--property=RuntimeMaxSec=2100','--property=KillMode=control-group','--property=TimeoutStopSec=2',
            '--property=OOMPolicy=stop','--working-directory='+str(root),
            '--setenv=PYTHONPATH='+str(root),'--setenv=OPENBLAS_NUM_THREADS=1','--setenv=OMP_NUM_THREADS=1',
            str(root/'.science/venv/bin/python'),'-m','tpu.science.routing_benchmark','--child',str(folder/'request.json')]
        if mode=='production':
            from .routing_resources import contract
            (folder/'source.py').write_text(source)
            save(folder/'request.json',dict(task='routing',root=str(root),work=request['work'],
                source=str(folder/'source.py'),routing_suite='full',resource_contract=contract()))
            cmd=cmd[:-4]+['-m','tpu.science.worker','--request',str(folder/'request.json'),
                '--result',request['result'],'--owner-pid',str(request['owner_pid']),
                '--owner-start',request['owner_start']]
        started=time.monotonic()
        try:
            with (folder/'unit.log').open('wb') as log:
                p=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=2130)
        finally:
            subprocess.run(['sudo','-n','systemctl','stop',unit],capture_output=True,timeout=15)
        if not (folder/'result.json').exists():
            if mode!='production':
                raise RuntimeError('benchmark child did not report: '+(folder/'unit.log').read_text(errors='replace')[-3000:])
            from .rewards import invalid
            from .routing_parallel import partial_metrics
            result=invalid('CPU task exited '+str(p.returncode)+' without result (timeout, resource limit, or worker failure)',phase='worker')
            result['stdout']=(folder/'unit.log').read_text(errors='replace')[-3000:]
            result['metrics'].update(partial_metrics(folder/'evaluation'),resource_contract=contract(),
                source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                grader_sha256=hashlib.sha256(b''.join(f.name.encode()+b'\0'+f.read_bytes()
                    for f in sorted((root/'tpu/science').glob('*.py')))).hexdigest())
            save(folder/'result.json',result)
        result=json.loads((folder/'result.json').read_text());result.update(mode=mode,label=label,
            seconds=seconds,unit_exit=p.returncode,queue_seconds=waited,outer_wall_seconds=time.monotonic()-started,
            unit=unit,host=__import__('socket').gethostname())
        # A stopped/collected unit must have no remaining processes or descendants.
        cg=Path('/sys/fs/cgroup/system.slice')/(unit+'.service')
        survivors=[]
        if cg.exists():
            for file in cg.rglob('cgroup.procs'):survivors.extend(file.read_text().split())
        result['remaining_pids']=survivors
        if survivors:raise RuntimeError('benchmark unit leaked descendants')
        save(folder/'result.json',result)
        # Preserve diagnostic evidence; discard reproducible compiler artifacts.
        import shutil
        for name in ('rust','target'):
            path=folder/'evaluation'/name
            if path.exists():shutil.rmtree(path)
        return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--child');p.add_argument('--panel');p.add_argument('--output');a=p.parse_args()
    if a.child:return child(json.loads(Path(a.child).read_text()))
    root=Path.cwd();out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    panel=json.loads(Path(a.panel).read_text());rank=int(os.environ['SKYPILOT_NODE_RANK'])
    source=panel['winner'] if rank in (0,1,2,7) else panel['timeout']
    mode={0:'parallel',1:'single',2:'legacy',3:'parallel',4:'single',5:'legacy',6:'parallel',7:'parallel'}[rank]
    if rank==6:source='RUST_CODE = "this deliberately does not compile"'
    if rank==0:
        r=evaluate_one(root,out,source,'parallel',120,'deadline-probe')
        print(json.dumps(dict(label='deadline-probe',correctness=r['correctness'],metrics=r['metrics'])),flush=True)
    label=f'rank-{rank}-{mode}'
    r=evaluate_one(root,out,source,mode,1800 if mode=='legacy' else 1900,label)
    print(json.dumps(dict(label=label,correctness=r['correctness'],reward=r['reward'],metrics=r['metrics'])),flush=True)
    save(out/'complete.json',dict(rank=rank,result=r))


if __name__=='__main__':main()
