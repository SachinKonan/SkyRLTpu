"""Finite lab-only watcher: preserve training, then run frozen normal audits."""
import glob
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import time
from launch_training import ROOT,CACHE,PYTHON,sha


def await_reports(items,timeout):
    start=time.monotonic();last=-1
    while True:
        done=0
        for unit,path in items:
            if path.is_file():done+=1;continue
            result=subprocess.run(['systemctl','show',unit,'-p','ActiveState','-p','Result','-p','LoadState'],text=True,capture_output=True).stdout
            if 'ActiveState=failed' in result or 'Result=failed' in result or 'LoadState=not-found' in result:
                log=subprocess.run(['journalctl','-u',unit,'-n','20','-o','cat','--no-pager'],text=True,capture_output=True).stdout
                raise RuntimeError(json.dumps(dict(unit=unit,expected=str(path),state=result,log=log)))
        if done!=last:
            print(json.dumps(dict(stage='waiting',complete=done,total=len(items),seconds=time.monotonic()-start)),flush=True);last=done
        if done==len(items):return
        if time.monotonic()-start>timeout:raise TimeoutError('Lab stage incomplete; inspect its own units.')
        time.sleep(10)


def await_free():
    for _ in range(12):
        r=subprocess.run(['sudo','fuser']+glob.glob('/dev/vfio/[0-9]*'),text=True,capture_output=True)
        if r.returncode==1 and not r.stdout.strip():return
        time.sleep(5)
    raise RuntimeError('TPUs remain occupied; do not override the owner.')


def launch(unit,chip,script,args):
    cmd=['sudo','systemd-run','--unit='+unit,'--uid=gcpuser','--collect',
         '--property=LimitMEMLOCK=infinity','--property=MemoryMax=80G','--property=MemorySwapMax=0',
         '--setenv=TPU_VISIBLE_CHIPS='+str(chip),'--setenv=TPU_CHIPS_PER_HOST_BOUNDS=1,1,1',
         '--setenv=TPU_HOST_BOUNDS=1,1,1','--setenv=OPENBLAS_NUM_THREADS=2',
         '--setenv=OMP_NUM_THREADS=2','--setenv=PYTHONUNBUFFERED=1',PYTHON,str(ROOT/'code/fair_r8'/script)]+args
    subprocess.run(cmd,check=True,capture_output=True,text=True)


def main():
    assert os.uname().nodename=='t1v-n-51b3bdd2-w-0'
    marker=ROOT/'post-training-start.json';assert not marker.exists(),'Audit prior post-training work before retrying.'
    marker.write_text(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),source_sha256=sha(__file__))))
    runs=[(16,0),(16,1),(32,0),(32,1)]
    await_reports([(f'fair-r8-train-w{w}-s{s}',ROOT/f'field-w{w}-s{s}/complete.json') for w,s in runs],2400)
    await_free()
    frozen={f'field-w{w}-s{s}':dict(checkpoint_sha256=sha(ROOT/f'field-w{w}-s{s}/best.msgpack'),
                                checkpoint=json.loads((ROOT/f'field-w{w}-s{s}/best.json').read_text()),
                                complete=json.loads((ROOT/f'field-w{w}-s{s}/complete.json').read_text())) for w,s in runs}
    (ROOT/'frozen-checkpoints.json').write_text(json.dumps(frozen,indent=2))
    with tarfile.open(ROOT/'training-001.tar.gz','w:gz') as archive:
        for name in [*frozen,'frozen-checkpoints.json','launch-001.json','post-training-start.json']:
            archive.add(ROOT/name,arcname=name)
    print(json.dumps(dict(stage='training_preserved',sha256=sha(ROOT/'training-001.tar.gz'))),flush=True)
    for chip,(w,s) in enumerate(runs):
        launch(f'fair-r8-num-w{w}-s{s}',chip,'numerical.py',
               ['--run',str(ROOT/f'field-w{w}-s{s}'),'--cache',str(CACHE),'--out',str(ROOT/f'numerical-w{w}-s{s}')])
    await_reports([(f'fair-r8-num-w{w}-s{s}',ROOT/f'numerical-w{w}-s{s}/report.json') for w,s in runs],1800)
    await_free()
    for w,s in runs:
        report=json.loads((ROOT/f'numerical-w{w}-s{s}/report.json').read_text())
        assert report['proceed64'],report
        assert report['checkpoint_sha256']==frozen[f'field-w{w}-s{s}']['checkpoint_sha256']
    launch('fair-r8-normal-quality',0,'audit_normal.py',
           ['--root',str(ROOT),'--old','/dev/shm/fair-r2-20261009','--cache',str(CACHE),'--out',str(ROOT/'normal-audit-001')])
    await_reports([('fair-r8-normal-quality',ROOT/'normal-audit-001/report.json')],1200)
    await_free()
    with tarfile.open(ROOT/'normal-study-001.tar.gz','w:gz') as archive:
        for name in [*frozen,*[f'numerical-w{w}-s{s}' for w,s in runs],'normal-audit-001',
                     'frozen-checkpoints.json','launch-001.json','post-training-start.json']:
            archive.add(ROOT/name,arcname=name)
    receipt=dict(kind='R8_COMPLETED_TRAINING_AND_NORMAL_ONLY_AUDITS',public_test_read=False,
                 sha256=sha(ROOT/'normal-study-001.tar.gz'),bytes=(ROOT/'normal-study-001.tar.gz').stat().st_size,
                 checkpoints=frozen,utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),no_vfio_owners=True)
    (ROOT/'normal-study-001.receipt.json').write_text(json.dumps(receipt,indent=2))
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
