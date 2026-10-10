"""Copy this experiment's artifacts from its four authorized hosts to durable GPFS."""
import argparse
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

CLUSTER = 'tpuswarm-v5p32-east5a-erdos-3242'
CONF = '/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh/' + CLUSTER
IPS = ['10.202.15.214', '10.202.15.216', '10.202.15.213', '10.202.15.215']
REMOTE = r'''
from pathlib import Path
import tarfile, sys
root=Path('/dev/shm/fair-ae-research')
allowed={'.py','.json','.jsonl','.npy','.npz','.msgpack','.zip','.md','.txt','.sh'}
paths=[]
for p in root.iterdir():
    if p.is_file() and p.suffix in allowed and p.name!='public_data.zip':
        paths.append(p)
    elif p.is_dir() and ((p/'config.json').exists() or p.name.startswith('code-v') or p.name in ['preliminary','final-components','final-exports']):
        for q in p.rglob('*'):
            if q.is_file() and q.suffix in allowed and '__pycache__' not in q.parts:
                # Never archive an in-progress density as a completed fitted model.
                if 'density' in q.parts:
                    density=next(x for x in q.parents if x.name=='density')
                    if not (density/'report.json').exists(): continue
                paths.append(q)
with tarfile.open(fileobj=sys.stdout.buffer,mode='w|gz') as tar:
    for p in sorted(set(paths)): tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
'''

def fetch(rank, dest):
    cmd='python3 -c '+shlex.quote(REMOTE)
    if rank:
        cmd='ssh -n -c aes128-gcm@openssh.com -F /dev/null -i ~/.ssh/sky-cluster-key -o StrictHostKeyChecking=no '+IPS[rank]+' '+shlex.quote(cmd)
    path=dest/f'host{rank}.tar.gz'
    for attempt in range(3):
        with path.open('wb') as out:
            result=subprocess.run(['ssh','-c','aes128-gcm@openssh.com','-F',CONF,'-o','BatchMode=yes',CLUSTER,cmd],stdout=out)
        if result.returncode==0:break
    result.check_returncode()
    result={'host':rank,'file':str(path),'bytes':path.stat().st_size,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
    print(json.dumps(result),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    dest=Path(a.out);dest.mkdir(parents=True,exist_ok=False)
    with concurrent.futures.ThreadPoolExecutor(4) as pool:
        records=list(pool.map(lambda rank:fetch(rank,dest),range(4)))
    (dest/'manifest.json').write_text(json.dumps(records,indent=2))
