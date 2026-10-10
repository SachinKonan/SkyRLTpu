"""Run only on the authorized erdos-3242 head, with no fleet operations."""
import glob
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import time

ROOT=Path('/dev/shm/fair-r8-20261009')
CACHE=Path('/dev/shm/fair-r2-20261009/field-cache')
PYTHON='/home/gcpuser/.cache/inference-farm-rq4-qwen35-east5a-v5p32-008/envs/serving/bin/python'


def sha(path):
    d=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):d.update(b)
    return d.hexdigest()


def main():
    assert os.uname().nodename=='t1v-n-51b3bdd2-w-0'
    target=ROOT/'launch-001.json';assert not target.exists(),'Audit an existing launch before retrying.'
    owners=subprocess.run(['sudo','fuser']+glob.glob('/dev/vfio/[0-9]*'),text=True,capture_output=True)
    assert owners.returncode==1 and not owners.stdout.strip(),owners
    free=os.statvfs(ROOT);free_bytes=free.f_bavail*free.f_frsize
    assert free_bytes>64*1024**3
    assert (CACHE/'clean.npy').is_file() and (CACHE/'normalization.npz').is_file()
    paths=[CACHE/'clean.npy',CACHE/'normalization.npz',ROOT/'code/fair_r2/field_flow.py',
           ROOT/'code/fair_r2/common.py',ROOT/'code/fair_r8/PLAN.json',ROOT/'code/fair_r8/launch_training.py']
    record=dict(kind='R8_NORMAL_ONLY_TRAINING_ON_AUTHORIZED_LAB',
                utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),host=os.uname().nodename,
                tmpfs_free_bytes=free_bytes,vfio_owners=owners.stdout.strip(),
                meminfo=Path('/proc/meminfo').read_text(),hashes={str(p):sha(p) for p in paths},
                versions={p:importlib.metadata.version(p) for p in ['jax','jaxlib','libtpu','flax','optax','numpy']},
                public_test_read=False,gcs_used=False,units=[])
    target.write_text(json.dumps(record,indent=2))
    for chip,(width,seed) in enumerate([(16,0),(16,1),(32,0),(32,1)]):
        name=f'field-w{width}-s{seed}';unit=f'fair-r8-train-w{width}-s{seed}'
        args=['sudo','systemd-run','--unit='+unit,'--uid=gcpuser','--collect',
              '--property=LimitMEMLOCK=infinity','--property=MemoryMax=80G','--property=MemorySwapMax=0',
              '--setenv=TPU_VISIBLE_CHIPS='+str(chip),'--setenv=TPU_CHIPS_PER_HOST_BOUNDS=1,1,1',
              '--setenv=TPU_HOST_BOUNDS=1,1,1','--setenv=OPENBLAS_NUM_THREADS=2',
              '--setenv=OMP_NUM_THREADS=2','--setenv=PYTHONUNBUFFERED=1',PYTHON,
              str(ROOT/'code/fair_r2/field_flow.py'),'fit','--cache',str(CACHE),
              '--out',str(ROOT/name),'--steps','16000','--batch','8','--width',str(width),
              '--seed',str(seed),'--val-count','128','--eval-every','500']
        result=subprocess.run(args,check=True,capture_output=True,text=True)
        record['units'].append(dict(chip=chip,width=width,seed=seed,unit=unit,output=str(ROOT/name),command=args))
        target.write_text(json.dumps(record,indent=2))
        print(json.dumps(dict(stage='launched',chip=chip,width=width,seed=seed,unit=unit)),flush=True)


if __name__=='__main__':main()
