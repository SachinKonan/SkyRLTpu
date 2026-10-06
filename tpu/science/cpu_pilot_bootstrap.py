"""Managed eight-host CPU pilot using the native workload's private Ray setup.

Runs under a pool lease on TPU hosts, but publishes zero Ray TPU resources.
Only this run's Ray temporary directory is retired; the pool Ray is untouched.
"""
import argparse
import fcntl
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
    if len(ips)!=8 or len(set(ips))!=8 or not 0<=rank<8:raise ValueError('expected eight TPU hosts')
    run='science-portfolio-v2-cpu-pilot-001'
    if not args.runtime_ready:
        lock=os.open(root/'owner.lock',os.O_CREAT|os.O_RDWR,0o600)
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);os.set_inheritable(lock,True)
        os.environ['SCIENCE_OWNER_FD']=str(lock)
        from tpu.swarm.ray_train.bootstrap import ensure_controller_runtime
        python=ensure_controller_runtime(Path.home()/'.cache/skyrl-ray-science')
        os.execv(str(python),[str(python),'-m','tpu.science.cpu_pilot_bootstrap','--runtime-ready'])
    os.fstat(int(os.environ['SCIENCE_OWNER_FD']))
    import ray
    from tpu.swarm.ray_train.bootstrap import stop_ray,check_ports_available
    ray_tmp=Path.home()/'.science-ray-v2-cpu-001'
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
        command=[str(Path(sys.executable).with_name('ray')),'start','--node-ip-address='+ips[rank],
            '--num-cpus=32','--resources={"TPU":0}','--object-store-memory=1073741824',
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
            while len([n for n in ray.nodes() if n['Alive']])!=8:
                if stopped or time.monotonic()>deadline:raise TimeoutError('missing CPU hosts')
                time.sleep(2)
            out=root/'pilot-results';out.mkdir(exist_ok=False)
            source_names=['seed_portfolio_v2_equal.py','seed_portfolio_v2_text.py','seed_portfolio_v2_xgb.py']
            cmd=[sys.executable,'-m','tpu.science.ray_pilot','--address',ips[0]+':'+str(port),
                 '--namespace',run,'--root',str(root),'--task','portfolio_v2','--all-nodes',
                 '--output',str(out/'references'),'--sources',*['tpu/science/'+n for n in source_names]]
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
                if code==0:
                    rows=[json.loads(x) for x in (out/'references/results.jsonl').read_text().splitlines()]
                    code=0 if len(rows)==24 and all(r['correctness']==1 for r in rows) else 1
                    (out/'summary.json').write_text(json.dumps(dict(exit_code=code,results=rows),indent=2)+'\n')
                archive=root/'pilot-results.tar.gz'
                subprocess.run(['tar','-czf',str(archive),'-C',str(root),'pilot-results'],check=True)
                subprocess.run(['gcloud','storage','cp',str(archive),os.environ['SCIENCE_RESULT_URI']],check=True,timeout=120)
                ray.get(status.finish.remote(code));published=True
            terminal,acks=ray.get(status.read.remote(rank),timeout=10)
            if terminal is not None:
                code=terminal;terminal_at=terminal_at or time.monotonic()
                if rank!=0 or acks==8 or time.monotonic()-terminal_at>30:break
            time.sleep(2)
        else:raise TimeoutError('CPU pilot interrupted or deadline exceeded')
    finally:
        if driver and driver.poll() is None:driver.terminate();driver.wait(timeout=20)
        ray.shutdown();stop_ray(ray_tmp)
        print(json.dumps(dict(event='cpu_pilot_stopped',rank=rank,exit_code=code)),flush=True)
    raise SystemExit(code)


if __name__=='__main__':main()
