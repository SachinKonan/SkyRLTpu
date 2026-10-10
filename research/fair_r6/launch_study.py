"""Run on the authorized erdos-3242 head only; no SkyPilot or GCS operations."""
import concurrent.futures
import glob
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

ROOT='/dev/shm/fair-r6-20261009'
PREVIOUS='/dev/shm/fair-r4-20261009'
PYTHON='/home/gcpuser/.cache/inference-farm-rq4-qwen35-east5a-v5p32-008/envs/serving/bin/python'
PEERS=[None,'10.202.15.216','10.202.15.213','10.202.15.215']
KEY='/home/gcpuser/.ssh/sky-cluster-key'
EXPECTED_NORMAL='d987e724e122e120f18993fee38572c47fc3e7722bbc668eecefb5690727537b'
EXPECTED_CLOSURE='140f06810177c7e7be2a61cf953badaf2cea7d67180a124e1b025bef6d6f7ce3'
WEIGHTS=['96dfa69ca5f5a137b2bd0f182a50aba6ccb9a28eec36098b59543a6330265013',
         '4a1c30c1342b1839025b4130e60509f44844349e10f7777d9a550caad6de2009']


def execute(ip,command):
    if ip is None:args=['bash','-c',command]
    else:args=['ssh','-n','-F','/dev/null','-i',KEY,'-o','StrictHostKeyChecking=no',
               '-o','BatchMode=yes','-o','ConnectTimeout=15',ip,command]
    return subprocess.run(args,check=True,text=True,capture_output=True).stdout


def main():
    assert os.uname().nodename=='t1v-n-51b3bdd2-w-0', 'This launcher is restricted to the handed-over machine.'
    target=Path(ROOT)/'launch-001.json';assert not target.exists(), 'Existing launch must be audited before retrying.'
    pilot_reports={}
    for seed in range(2):
        pilot=json.loads((Path(ROOT)/f'pilot-s{seed}/report.json').read_text())
        refined=json.loads((Path(ROOT)/f'refine-s{seed}/report.json').read_text())
        assert max(pilot['pointwise_errors'])==0
        assert pilot['checkpoint_sha256']==WEIGHTS[seed]==refined['checkpoint_sha256']
        for name in ('energy','ad','ad_shape','psd_cv','correlation_energy'):
            assert refined['difference64_128'][name]['median_over_normal_sd']<.003
        pilot_reports[str(seed)]=dict(pilot=pilot,refinement=refined)
    for ip in PEERS[1:]:
        execute(ip,'mkdir -p '+ROOT+'/code')
        # Copy only this experiment's small code and normal-data cache, within this lab machine.
        subprocess.run(['scp','-q','-r','-F','/dev/null','-i',KEY,'-o','BatchMode=yes',
                        '-o','StrictHostKeyChecking=no',ROOT+'/normals',ROOT+'/code-evaluate.tar.gz',
                        ip+':'+ROOT+'/'],check=True)
        execute(ip,'tar -xzf '+ROOT+'/code-evaluate.tar.gz -C '+ROOT+'/code && '
                   'ln -s '+PREVIOUS+'/code/fair_r2 '+ROOT+'/code/fair_r2 && '
                   'ln -s '+PREVIOUS+'/code/fair_ae '+ROOT+'/code/fair_ae')
        print(json.dumps(dict(stage='staged',ip=ip)),flush=True)
    check_code='''import json,glob,hashlib,subprocess,os
paths={paths!r}
checks={{}}
for p in paths:
 d=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(16*1024*1024),b''):d.update(b)
 checks[p]=d.hexdigest()
gpu=subprocess.run(['sudo','fuser']+glob.glob('/dev/vfio/[0-9]*'),capture_output=True,text=True)
print(json.dumps(dict(host=os.uname().nodename,hashes=checks,vfio_pids=gpu.stdout.strip())))
'''.format(paths=[ROOT+'/normals/observed.npy',PREVIOUS+'/closure-cache/observed.npy',
                  PREVIOUS+'/field-s0/best.msgpack',PREVIOUS+'/field-s1/best.msgpack',
                  ROOT+'/code/fair_r6/noise_statistics.py',ROOT+'/code/fair_r6/field_latents.py'])
    def audit(ip):return json.loads(execute(ip,'python3 -c '+shlex.quote(check_code)))
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        audits=list(pool.map(audit,PEERS))
    for audit in audits:
        assert not audit['vfio_pids'],audit
        hashes=audit['hashes'];assert hashes[ROOT+'/normals/observed.npy']==EXPECTED_NORMAL
        assert hashes[PREVIOUS+'/closure-cache/observed.npy']==EXPECTED_CLOSURE
        for seed in range(2):assert hashes[PREVIOUS+f'/field-s{seed}/best.msgpack']==WEIGHTS[seed]
        assert hashes==audits[0]['hashes']
    record=dict(kind='R6_LAB_ONLY_FROZEN_STATISTICS_LAUNCH',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                machine='tpuswarm-v5p32-east5a-erdos-3242',steps=64,models=2,shards_per_model=8,
                rows_per_model=16160,normal_and_reused_synthetic_only=True,public_test_read=False,
                numerical_decision='64 steps: median statistic changes against 128 steps are below 0.3 percent of the 16-map normal-pilot SD for both models.',
                pilots=pilot_reports,host_audits=audits,units=[])
    target.write_text(json.dumps(record,indent=2))
    for host,ip in enumerate(PEERS):
        for chip in range(4):
            seed=chip//2;shard=2*host+chip%2;unit=f'fair-r6-h{host}-c{chip}'
            args=['sudo','systemd-run','--unit='+unit,'--uid=gcpuser','--collect',
                  '--property=LimitMEMLOCK=infinity','--property=MemoryMax=80G','--property=MemorySwapMax=0',
                  '--setenv=TPU_VISIBLE_CHIPS='+str(chip),'--setenv=TPU_CHIPS_PER_HOST_BOUNDS=1,1,1',
                  '--setenv=TPU_HOST_BOUNDS=1,1,1','--setenv=OPENBLAS_NUM_THREADS=2',
                  '--setenv=OMP_NUM_THREADS=2','--setenv=PYTHONUNBUFFERED=1',PYTHON,
                  ROOT+'/code/fair_r6/evaluate_cache.py','--run',PREVIOUS+f'/field-s{seed}',
                  '--normals',ROOT+'/normals','--closure',PREVIOUS+'/closure-cache',
                  '--out',ROOT+f'/field-s{seed}-{shard}','--shard',str(shard),'--shards','8','--steps','64']
            execute(ip,shlex.join(args))
            record['units'].append(dict(host=host,ip=ip,chip=chip,seed=seed,shard=shard,unit=unit))
            target.write_text(json.dumps(record,indent=2))
            print(json.dumps(dict(stage='submitted',unit=unit,seed=seed,shard=shard)),flush=True)
    print(json.dumps(dict(stage='launch_complete',units=len(record['units']))),flush=True)


if __name__=='__main__':main()
