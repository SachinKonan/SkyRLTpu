"""One Ray-admitted placement case, within a verified systemd cgroup."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import resource
import signal
import subprocess
import sys
import time

from .cgroup_limits import envelope, metrics
from .isolation import command, python_mounts
from .rewards import invalid, valid
from .worker import watch_owner

from .challenge_contract import CASES
from .placement_slots import device_paths, tpu_environment, assigned_chip


def evaluate(request):
    root = Path(request['root']); work = Path(request['work']); work.mkdir()
    case = request['case']
    if case not in CASES: raise ValueError('unknown placement case')
    python = root/'.science/candidate-venv/bin/python'
    source = Path(request['source'])
    if source.stat().st_size > 1024**2: raise ValueError('oversized source')
    problem = root/'.science/placement-inputs'/f'{case}-problem.npz'
    tpu = request.get('backend') == 'tpu'
    env = {'OPENBLAS_NUM_THREADS':'4','OMP_NUM_THREADS':'4','MKL_NUM_THREADS':'4',
           'NUMEXPR_NUM_THREADS':'4','XDG_CACHE_HOME':'/tmp/cache'}
    mounts = python_mounts(python)+[(source,'/candidate.py'),(problem,'/problem.npz'),
        (root/'tpu/science/challenge_candidate_child.py','/runner.py')]
    if tpu:
        chip = assigned_chip({'TPU': request.get('tpu_ids', [])})
        devices = device_paths(chip, request.get('accelerator', 'tpu-v4-64'))
        # Check the assigned accelerator node, not the shared VFIO control node.
        busy = subprocess.run(['sudo','-n','fuser',str(devices[0])],capture_output=True,timeout=10)
        if busy.returncode != 1 or busy.stdout.strip() or busy.stderr.strip():
            raise RuntimeError('candidate chip is occupied or ownership check failed')
        mounts += [('/sys','/sys'),('/etc/hosts','/etc/hosts')]
        env.update(tpu_environment(chip, request.get('accelerator', 'tpu-v4-64'), isolated=True))
    cmd = command([str(python),'/runner.py','170','tpu' if tpu else 'cpu'],
                  readonly=mounts,writable=[(work,'/output')],env=env)
    if tpu:
        at = cmd.index('--dev')+2
        cmd[at:at] = [arg for device in devices for arg in ('--dev-bind',str(device),str(device))] + ['--tmpfs','/dev/shm']
    def constrain():
        # libtpu maps device memory virtually. Actual process-tree RAM is bounded
        # by the verified cgroup rather than RLIMIT_AS in the TPU profile.
        if not tpu: resource.setrlimit(resource.RLIMIT_AS,(16*1024**3,)*2)
        resource.setrlimit(resource.RLIMIT_FSIZE,(2*1024**2,)*2)
        resource.setrlimit(resource.RLIMIT_CORE,(0,0))
        resource.setrlimit(resource.RLIMIT_NOFILE,(1024,1024))
        resource.setrlimit(resource.RLIMIT_CPU,(725,725))
    candidate_started_unix = time.time()
    started = time.monotonic()
    with (work/'candidate.log').open('wb') as log:
        proc = subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,
                                start_new_session=True,preexec_fn=constrain,env={'PATH':'/usr/bin:/bin'})
        try:
            code = proc.wait(timeout=180)
        finally:
            # Kill any detached work inside the child's PID namespace on exit.
            if proc.poll() is None: os.killpg(proc.pid,signal.SIGKILL); proc.wait()
    candidate_finished_unix = time.time()
    candidate_seconds = time.monotonic()-started
    if code: raise RuntimeError(f'candidate exited {code}: '+(work/'candidate.log').read_text(errors='replace')[-2000:])
    cmd = [str(root/'.science/venv/bin/python'),'-m','tpu.science.challenge_score_child',
           '--root',str(root),'--case',case,'--positions',str(work/'positions.npy'),
           '--result',str(work/'score.json')]
    with (work/'grader.log').open('wb') as log:
        subprocess.run(cmd,check=True,timeout=90,stdout=log,stderr=subprocess.STDOUT,
                       env=dict(os.environ,OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
    scores = json.loads((work/'score.json').read_text())
    cost = scores['proxy_cost']
    return valid(max(1e-6,1/(1+cost)), dict(**scores,
                 candidate_wall_seconds=candidate_seconds,candidate_started_unix=candidate_started_unix,
                 candidate_finished_unix=candidate_finished_unix,case=case,backend='tpu' if tpu else 'cpu',
                 candidate_device=json.loads((work/'child.json').read_text())))


def main():
    p=argparse.ArgumentParser();p.add_argument('--request',required=True);p.add_argument('--result',required=True)
    p.add_argument('--owner-pid',type=int,required=True);p.add_argument('--owner-start',required=True)
    args=p.parse_args();watch_owner(args.owner_pid,args.owner_start)
    group,_=envelope(16)
    request=json.loads(Path(args.request).read_text());started=time.monotonic()
    try: result=evaluate(request)
    except Exception as exc: result=invalid(f'{type(exc).__name__}: {exc}')
    result['metrics'].update(allocation_memory=metrics(group),worker_seconds=time.monotonic()-started,
                             cpu_affinity=sorted(os.sched_getaffinity(0)),case=request['case'])
    for name in ['candidate.log','grader.log','child.json']:
        f=Path(request['work'])/name
        if f.exists():result['metrics'][name]=f.read_text(errors='replace')[-4000:]
    Path(args.result).write_text(json.dumps(result,allow_nan=False,indent=2)+'\n')


if __name__=='__main__':main()
