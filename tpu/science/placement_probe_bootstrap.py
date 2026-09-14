"""Managed placement probe using the native workload's private Ray setup.

Runs under a pool lease on TPU hosts, but publishes zero Ray TPU resources.
Only this run's Ray temporary directory is retired; the pool Ray is untouched.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def main():
    p=argparse.ArgumentParser();p.add_argument('--runtime-ready',action='store_true');args=p.parse_args()
    root=Path.cwd().resolve();ips=os.environ['SKYPILOT_NODE_IPS'].split();rank=int(os.environ['SKYPILOT_NODE_RANK'])
    accelerator=os.environ.get('SCIENCE_ACCELERATOR','tpu-v4-64')
    hosts={'tpu-v4-64':8,'tpu-v5p-32':4}.get(accelerator)
    if hosts is None or len(ips)!=hosts or len(set(ips))!=hosts or not 0<=rank<hosts:
        raise ValueError('placement probe host count does not match accelerator')
    from .placement_slots import chips_from_env, host_resources
    chips=chips_from_env(os.environ)
    grading_ranks=[int(r) for r in os.environ.get('PLACEMENT_TPU_RANKS','5,6,7' if hosts==8 else '3').split(',')]
    if not grading_ranks or len(set(grading_ranks))!=len(grading_ranks) or any(r not in range(hosts) for r in grading_ranks):
        raise ValueError('invalid grading host ranks')
    run=os.environ.get('SCIENCE_RUN_ID','science-placement-jax-probe-002')
    if not args.runtime_ready:
        lock=os.open(root/'owner.lock',os.O_CREAT|os.O_RDWR,0o600)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.set_inheritable(lock,True)
        os.environ['SCIENCE_OWNER_FD']=str(lock)
        from tpu.swarm.ray_train.bootstrap import ensure_controller_runtime
        python=ensure_controller_runtime(Path.home()/'.cache/skyrl-ray-science')
        os.execv(str(python),[str(python),'-m','tpu.science.placement_probe_bootstrap','--runtime-ready'])
    os.fstat(int(os.environ['SCIENCE_OWNER_FD']))
    import ray
    from tpu.swarm.ray_train.bootstrap import stop_ray,check_ports_available
    # Ray appends a session name and Unix socket name (Linux limit: 107 bytes).
    # Hash the run ID so the directory remains short and private to this run.
    ray_tmp=Path('/tmp')/('plc-'+str(os.getuid())+'-'+hashlib.sha256(run.encode()).hexdigest()[:12])
    socket_path=ray_tmp/'session_2026-09-14_00-00-00_999999_9999999'/'sockets'/'plasma_store'
    if len(os.fsencode(socket_path))>107:
        raise ValueError('placement Ray Unix socket path is too long')
    ray_tmp.mkdir(mode=0o700,exist_ok=True)
    if ray_tmp.is_symlink() or ray_tmp.stat().st_uid!=os.getuid():
        raise RuntimeError('placement Ray directory is not owned by this user')
    port=20679
    ports=[port,20680,20681,20682,20683,20684,20685,20686,20687]
    stopped=False
    def stop(*_):
        nonlocal stopped
        stopped=True
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    driver=None;code=1
    try:
        stop_ray(ray_tmp);check_ports_available(ports)
        os.environ.update(RAY_ADDRESS=ips[0]+':'+str(port),RAY_NAMESPACE=run,
            RAY_TMPDIR=str(ray_tmp),RAY_USAGE_STATS_ENABLED='0',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
        os.environ['TPU_VISIBLE_CHIPS'] = ','.join(map(str,chips)) if rank in grading_ranks else ''
        command=[str(Path(sys.executable).with_name('ray')),'start','--node-ip-address='+ips[rank],
            '--num-cpus=32','--resources='+json.dumps(host_resources(chips) if rank in grading_ranks else {'TPU':0}),'--object-store-memory=1073741824',
            '--object-manager-port=20682','--node-manager-port=20683',
            '--dashboard-agent-listen-port=20684','--dashboard-agent-grpc-port=20685',
            '--runtime-env-agent-port=20686','--metrics-export-port=20687',
            '--min-worker-port=20700','--max-worker-port=20799','--disable-usage-stats']
        if rank==0:command+=['--head','--port='+str(port),'--dashboard-port=20680',
                             '--ray-client-server-port=20681','--temp-dir='+str(ray_tmp)]
        else:
            deadline=time.monotonic()+180
            while True:
                try:
                    with socket.create_connection((ips[0],port),timeout=2):break
                except OSError:
                    if stopped or time.monotonic()>deadline:raise TimeoutError('head startup')
                    time.sleep(2)
            command+=['--address='+ips[0]+':'+str(port)]
        subprocess.run(command,check=True,timeout=180)
        ray.init(address=ips[0]+':'+str(port),namespace=run)
        if rank==0:
            @ray.remote(num_cpus=0)
            class Terminal:
                def __init__(self):self.code=None;self.acks=set()
                def finish(self,code):self.code=code
                def read(self,rank):
                    if self.code is not None:self.acks.add(rank)
                    return self.code,len(self.acks)
            status=Terminal.options(name='cpu-pilot-status',lifetime='detached').remote()
            deadline=time.monotonic()+180
            while len([n for n in ray.nodes() if n['Alive']])!=hosts:
                if stopped or time.monotonic()>deadline:raise TimeoutError('missing CPU hosts')
                time.sleep(2)
            out=root/'pilot-results';out.mkdir(exist_ok=False)
            cmd=[sys.executable,'-m','tpu.science.placement_probe_driver',
                 '--address',ips[0]+':'+str(port),'--namespace',run,
                 '--root',str(root),'--output',str(out),'--accelerator',accelerator,
                 '--expected-slots',str(len(grading_ranks)*len(chips))]
            driver=subprocess.Popen(cmd)
        else:
            deadline=time.monotonic()+180
            while True:
                try:status=ray.get_actor('cpu-pilot-status',namespace=run);break
                except ValueError:
                    if stopped or time.monotonic()>deadline:raise TimeoutError('head status missing')
                    time.sleep(2)
        deadline=time.monotonic()+1500;published=False;terminal_at=None
        while not stopped and time.monotonic()<deadline:
            if rank==0 and driver.poll() is not None and not published:
                code=driver.returncode
                archive=root/'pilot-results.tar.gz'
                subprocess.run(['tar','-czf',str(archive),'-C',str(root),'pilot-results'],check=True)
                subprocess.run(['gcloud','storage','cp',str(archive),os.environ['SCIENCE_RESULT_URI']],check=True,timeout=120)
                ray.get(status.finish.remote(code));published=True
            terminal,acks=ray.get(status.read.remote(rank),timeout=10)
            if terminal is not None:
                code=terminal;terminal_at=terminal_at or time.monotonic()
                if rank!=0 or acks==hosts or time.monotonic()-terminal_at>30:break
            time.sleep(2)
        else:raise TimeoutError('CPU pilot interrupted or deadline exceeded')
    finally:
        if driver and driver.poll() is None:driver.terminate();driver.wait(timeout=20)
        ray.shutdown();stop_ray(ray_tmp)
        print(json.dumps(dict(event='cpu_pilot_stopped',rank=rank,exit_code=code)),flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
