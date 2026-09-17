import concurrent.futures,io,json,subprocess,tarfile
from pathlib import Path
p=Path('.science/placement-cpu-20260917');base='tpuswarm-v6e32-east5b-qwen35-3744';cfg='/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh/'+base
root='/home/gcpuser/.cache/science-placement-cpu-regrade-20260917';old='/home/gcpuser/.cache/skyrl-ray-code/7ab6a72ce38b94369298f3c0e940722cbc2211cb3b15a7c55e14029d45278fbb';python='/home/gcpuser/.cache/science-placement-v6e-muse-xplace-bootstrap-l2-001/envs/controller/bin/python'
archive=p/'regrade-code.tar.gz'
with tarfile.open(archive,'w:gz') as tar:
 for f in Path('tpu/science').glob('*.py'):tar.add(f,arcname=str(f),recursive=False)
 for name in ['run_remote_regrade.py','selected-seeds.json']:tar.add(p/name,arcname=name)
setup=f'''from pathlib import Path
import json,os,sys,tarfile
root=Path({root!r});old=Path({old!r});root.mkdir(exist_ok=False)
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz') as tar:tar.extractall(root,filter='data')
(root/'.science').mkdir()
for name in ['venv','candidate-venv','placement-inputs','challenge-probe','ready.json']:
 source=old/'.science'/name
 if not source.exists():raise RuntimeError('missing '+str(source))
 (root/'.science'/name).symlink_to(source)
print(json.dumps(dict(host=__import__('socket').gethostname(),root=str(root),ready=True)))
'''
def host(i):
 alias=base+('' if i==0 else f'-worker{i}');cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','LogLevel=ERROR','-F',cfg,alias]
 r=subprocess.run(cmd+['python3 -c '+__import__('shlex').quote(setup)],input=archive.read_bytes(),capture_output=True,timeout=60)
 (p/f'host-{i}-setup.log').write_bytes(r.stdout+r.stderr);r.check_returncode()
 with (p/f'host-{i}-regrade.log').open('w') as f:
  r=subprocess.run(cmd+['cd '+root+' && PYTHONPATH='+root+' '+python+' run_remote_regrade.py '+str(i)],stdout=f,stderr=subprocess.STDOUT,timeout=3600)
 print('host',i,'exit',r.returncode,flush=True)
 r.check_returncode()
 r=subprocess.run(cmd+['tar -czf - -C '+root+' regrade'],capture_output=True,timeout=90);r.check_returncode();(p/f'host-{i}-results.tar.gz').write_bytes(r.stdout)
 dest=p/f'host-{i}';dest.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(r.stdout)) as tar:tar.extractall(dest,filter='data')
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:list(pool.map(host,range(8)))
