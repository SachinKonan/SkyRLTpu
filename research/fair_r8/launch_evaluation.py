"""Run the registered, matched likelihood comparison on erdos-3242 only."""
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time
from launch_training import ROOT,PYTHON,sha

PEERS=[None,'10.202.15.216','10.202.15.213','10.202.15.215']
KEY='/home/gcpuser/.ssh/sky-cluster-key'


def execute(ip,command):
    args=['bash','-c',command] if ip is None else ['ssh','-n','-F','/dev/null','-i',KEY,
        '-o','StrictHostKeyChecking=no','-o','BatchMode=yes','-o','ConnectTimeout=15',ip,command]
    return subprocess.run(args,check=True,text=True,capture_output=True).stdout


def main():
    assert os.uname().nodename=='t1v-n-51b3bdd2-w-0'
    receipt=json.loads((ROOT/'normal-study-001.receipt.json').read_text())
    assert receipt['no_vfio_owners']
    target=ROOT/'evaluation-launch-001.json';assert not target.exists(),'Audit a prior launch before retrying.'
    quality=json.loads((ROOT/'normal-audit-001/report.json').read_text())
    means={w:sum(quality['records'][f'w{w}-s{s}']['uniform_t_mse'] for s in range(2))/2 for w in (16,32)}
    width=min(means,key=means.get)
    selection=dict(kind='NORMAL_ONLY_ARCHITECTURE_SELECTION_NOT_CANDIDATE_PROMOTION',selected_width=width,
        uniform_t_two_seed_mean_mse=means,quality_sha256=sha(ROOT/'normal-audit-001/report.json'),
        evaluation_plan_sha256=sha(ROOT/'code/fair_r8/EVALUATION_PLAN.json'))
    (ROOT/'selection.json').write_text(json.dumps(selection,indent=2))
    models=['old-s0','old-s1','new-s0','new-s1'];source={}
    for name in models:
        seed=int(name[-1]);path=(ROOT/f'field-w{width}-s{seed}' if name.startswith('new') else
                              Path('/dev/shm/fair-r2-20261009')/f'field-s{seed}')
        if name.startswith('new'):
            report=json.loads((ROOT/f'numerical-w{width}-s{seed}/report.json').read_text())
            assert report['proceed64'] and report['checkpoint_sha256']==sha(path/'best.msgpack')
        destination=ROOT/'models'/name;destination.mkdir(parents=True,exist_ok=False)
        for filename in ('best.msgpack','normalization.npz','config.json','complete.json','best.json'):
            shutil.copy2(path/filename,destination/filename)
        source[name]=dict(source=str(path),sha256=sha(destination/'best.msgpack'))
    cache=ROOT/'evaluation-cache';manifest=json.loads((cache/'manifest.json').read_text())
    assert manifest['public_test_read'] is False and manifest['shape'][0]==13440
    assert manifest['observations_sha256']=='80d14c05faaf994eea41c48caa1f2ea6229af2d9d6870f38d42212cd94d82bdb'
    with tarfile.open(ROOT/'likelihood-code.tar.gz','w:gz') as f:
        for name in ('field_flow.py','field_evaluate.py','field_benchmark.py','common.py'):
            f.add(ROOT/'code/fair_r2'/name,arcname='fair_r2/'+name)
    for ip in PEERS[1:]:
        execute(ip,'mkdir -p '+str(ROOT/'code'))
        subprocess.run(['scp','-q','-r','-F','/dev/null','-i',KEY,'-o','BatchMode=yes',
                        '-o','StrictHostKeyChecking=no',str(cache),str(ROOT/'models'),str(ROOT/'likelihood-code.tar.gz'),
                        ip+':'+str(ROOT)+'/'],check=True)
        execute(ip,'tar -xzf '+str(ROOT/'likelihood-code.tar.gz')+' -C '+str(ROOT/'code'))
        print(json.dumps(dict(stage='staged',ip=ip)),flush=True)
    files=[cache/'observed.npy',cache/'source_ids.npy',cache/'manifest.json']
    files+=[ROOT/'models'/name/'best.msgpack' for name in models]
    files+=[ROOT/'code/fair_r2'/name for name in ('field_flow.py','field_evaluate.py','field_benchmark.py','common.py')]
    code='''import hashlib,json,glob,subprocess,os
hashes={{}}
for path in {paths!r}:
 d=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):d.update(b)
 hashes[path]=d.hexdigest()
p=subprocess.run(['sudo','fuser']+glob.glob('/dev/vfio/[0-9]*'),text=True,capture_output=True)
s=os.statvfs('/dev/shm');r=os.statvfs('/')
print(json.dumps(dict(host=os.uname().nodename,hashes=hashes,vfio_owners=p.stdout.strip(),vfio_returncode=p.returncode,
 tmpfs_free_bytes=s.f_bavail*s.f_frsize,root_free_bytes=r.f_bavail*r.f_frsize)))
'''.format(paths=[str(x) for x in files])
    def audit(ip):return json.loads(execute(ip,'python3 -c '+shlex.quote(code)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:audits=list(pool.map(audit,PEERS))
    for i,record in enumerate(audits):
        assert record['host']==f't1v-n-51b3bdd2-w-{i}'
        assert record['vfio_returncode']==1 and not record['vfio_owners']
        assert record['hashes']==audits[0]['hashes']
        assert record['hashes'][str(cache/'observed.npy')]==manifest['observations_sha256']
        assert record['tmpfs_free_bytes']>48*1024**3 and record['root_free_bytes']>20*1024**3
        for name in models:assert record['hashes'][str(ROOT/'models'/name/'best.msgpack')]==source[name]['sha256']
    record=dict(kind='R8_MATCHED_FROZEN_LIKELIHOOD_LAB_LAUNCH',
                utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),steps=64,batch=32,
                models=source,selection=selection,normal_and_reused_synthetic_only=True,public_test_read=False,
                host_audits=audits,units=[])
    target.write_text(json.dumps(record,indent=2))
    for host,(ip,name) in enumerate(zip(PEERS,models)):
        for chip in range(4):
            unit=f'fair-r8-eval-{name}-{chip}'
            args=['sudo','systemd-run','--unit='+unit,'--uid=gcpuser','--collect',
                  '--property=LimitMEMLOCK=infinity','--property=MemoryMax=80G','--property=MemorySwapMax=0',
                  '--setenv=TPU_VISIBLE_CHIPS='+str(chip),'--setenv=TPU_CHIPS_PER_HOST_BOUNDS=1,1,1',
                  '--setenv=TPU_HOST_BOUNDS=1,1,1','--setenv=OPENBLAS_NUM_THREADS=2',
                  '--setenv=OMP_NUM_THREADS=2','--setenv=PYTHONUNBUFFERED=1',PYTHON,
                  str(ROOT/'code/fair_r2/field_benchmark.py'),'score','--run',str(ROOT/'models'/name),
                  '--cache',str(cache),'--out',str(ROOT/f'eval-{name}-{chip}'),
                  '--batch','32','--ode-steps','64','--shard',str(chip),'--shards','4']
            execute(ip,shlex.join(args))
            record['units'].append(dict(host=host,ip=ip,model=name,chip=chip,unit=unit))
            target.write_text(json.dumps(record,indent=2));print(json.dumps(dict(stage='launched',model=name,chip=chip,host=host)),flush=True)


if __name__=='__main__':main()
