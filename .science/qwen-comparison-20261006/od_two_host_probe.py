"""Probe only two idle assigned hosts; never initialize the full occupied slice."""
import json,time,os
from concurrent.futures import ThreadPoolExecutor
from od_deploy import ssh,OUT
from launch_watch import save
ALL=[('34.58.61.190','qcmp-qwen38-qubit-pwc05-s1-shared-v464-wd-20261007'),('34.136.9.17','qcmp-qwen38-circuit-pwc05-s0-shared-v464-ram96-wd-20261007')]
ALL += [('34.122.223.241','qcmp-qwen38-ac1-pwc05-s0-east5-15steps-20261010'),('35.222.180.160','qcmp-qwen38-erdos-pwc05-s0-cap512-20261007')]
SELECT=list(map(int,os.environ.get('PROBE_HOSTS','0,1').split(',')))
PAIRS=[ALL[i] for i in SELECT]
INTERNAL=['10.128.1.94','10.128.1.95','10.128.1.91','10.128.1.93']
IPS=[INTERNAL[i] for i in SELECT]
TAG='-'.join(map(str,SELECT))

BOUNDS=os.environ.get('PROBE_BOUNDS','1,1,2')
program="""import jax,json,socket
import jax.numpy as jnp
from jax import lax
jax.distributed.initialize(coordinator_address='COORDINATOR',num_processes=2,process_id=RANK)
records={'process_id':RANK,'host':socket.gethostname(),'coords':[list(d.coords) for d in jax.local_devices()],'global_devices':len(jax.devices())}
x=jax.pmap(lambda x:lax.psum(x,'d'),axis_name='d')(jnp.ones(4))
records['all_reduce']=list(map(float,x));assert records['all_reduce']==[8.0]*4
print('PROBE_RESULT '+json.dumps(records),flush=True)
jax.distributed.shutdown()
"""
def run(item):
 rank,(ip,run_id)=item
 cmd=r'''import subprocess,json
from pathlib import Path
run_id=RUN
python=Path.home()/'.cache'/run_id/'envs/topology/bin/python'
assert python.is_file()
audit=subprocess.run(['sudo','-n','fuser',*map(str,Path('/dev/vfio').glob('[0-9]*'))],capture_output=True,text=True)
assert audit.returncode==1 and not audit.stdout.strip(), 'TPU still owned: '+audit.stdout
env=['JAX_PLATFORMS=tpu','TPU_PROCESS_BOUNDS=PROBE_SHAPE','TPU_CHIPS_PER_PROCESS_BOUNDS=2,2,1','TPU_PROCESS_ADDRESSES=PROBE_ADDRESSES','TPU_PROCESS_PORT=24804','CLOUD_TPU_TASK_ID=RANK','TPU_VISIBLE_CHIPS=0,1,2,3','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1']
p=subprocess.run(['sudo','-n','systemd-run','--quiet','--wait','--pipe','--collect','--unit=skyrl-two-host-probe-20261013','--uid=2021','--gid=2021','--property=RuntimeMaxSec=300','--property=KillMode=control-group','--property=LimitMEMLOCK=infinity','env',*env,str(python),'-c',PROGRAM],capture_output=True,text=True,timeout=330)
print(json.dumps({'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}))
'''.replace('PROBE_SHAPE',BOUNDS).replace('PROBE_ADDRESSES',','.join(ip+':24804' for ip in IPS)).replace('RUN',repr(run_id)).replace('RANK',str(rank)).replace('PROGRAM',repr(program.replace('RANK',str(rank)).replace('COORDINATOR',IPS[0]+':24807')))
 try:d=json.loads(ssh(ip,cmd,timeout=350))
 except Exception as e:d={'error':str(e)}
 save(OUT/f'two-host-probe-{TAG}-{BOUNDS}-{rank}.json',d)
 result={'rank':rank,'ip':ip,'returncode':d.get('returncode'),'error':d.get('error'),'records':[json.loads(l.removeprefix('PROBE_RESULT ')) for l in d.get('stdout','').splitlines() if l.startswith('PROBE_RESULT ')]}
 print(json.dumps(result),flush=True);return result
with ThreadPoolExecutor(2) as p:results=list(p.map(run,enumerate(PAIRS)))
save(OUT/f'two-host-probe-{TAG}-{BOUNDS}.json',results)
