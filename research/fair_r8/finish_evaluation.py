"""Finite watcher and artifact collection for the registered lab comparison."""
import concurrent.futures
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time
from launch_training import ROOT,sha
from launch_evaluation import PEERS,KEY,execute


def main():
    assert os.uname().nodename=='t1v-n-51b3bdd2-w-0'
    launch=json.loads((ROOT/'evaluation-launch-001.json').read_text());assert len(launch['units'])==16
    start=time.monotonic();last=-1
    code='''import pathlib,json,subprocess
r=pathlib.Path({root!r});name={name!r};rows=[]
for chip in range(4):
 unit=f'fair-r8-eval-{{name}}-{{chip}}';p=r/f'eval-{{name}}-{{chip}}';record=dict(model=name,chip=chip)
 if (p/'complete.json').exists():record.update(state='complete',result=json.loads((p/'complete.json').read_text()))
 else:
  state=subprocess.run(['systemctl','show',unit,'-p','ActiveState','-p','Result','-p','LoadState'],text=True,capture_output=True).stdout
  record['state']='failed' if any(x in state for x in ['ActiveState=failed','Result=failed','LoadState=not-found']) else 'running'
  if (p/'progress.json').exists():record['progress']=json.loads((p/'progress.json').read_text())
  if record['state']=='failed':record['log']=subprocess.run(['journalctl','-u',unit,'-n','15','-o','cat','--no-pager'],text=True,capture_output=True).stdout
 rows.append(record)
print(json.dumps(rows))
'''
    names=['old-s0','old-s1','new-s0','new-s1']
    def poll(i):
        # These are read-only status requests. Retry a transient SSH disconnect
        # without repeating launches or touching an active scoring process.
        for attempt in range(3):
            try:
                return json.loads(execute(PEERS[i],'python3 -c '+shlex.quote(code.format(root=str(ROOT),name=names[i]))))
            except subprocess.CalledProcessError as error:
                if error.returncode!=255 or attempt==2:raise
                print(json.dumps(dict(stage='status_ssh_retry',host=i,attempt=attempt+1,
                                      stderr=error.stderr)),flush=True)
                time.sleep(5)
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:rows=sum(list(pool.map(poll,range(4))),[])
        failed=[r for r in rows if r['state']=='failed'];assert not failed,failed
        done=sum(r['state']=='complete' for r in rows)
        progress=ROOT/'evaluation-progress.json'
        progress.with_suffix('.tmp').write_text(json.dumps(dict(utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),rows=rows),indent=2))
        progress.with_suffix('.tmp').replace(progress)
        if done!=last:
            print(json.dumps(dict(stage='waiting',complete=done,total=16,seconds=time.monotonic()-start)),flush=True);last=done
        if done==16:break
        if time.monotonic()-start>3600:raise TimeoutError('Likelihood study remains incomplete; inspect its own units.')
        time.sleep(15)
    for ip,name in zip(PEERS[1:],names[1:]):
        for chip in range(4):
            subprocess.run(['scp','-q','-r','-F','/dev/null','-i',KEY,'-o','StrictHostKeyChecking=no',
                            '-o','BatchMode=yes',ip+':'+str(ROOT/f'eval-{name}-{chip}'),str(ROOT)+'/'],check=True)
    for name in names:
        for chip in range(4):
            p=ROOT/f'eval-{name}-{chip}';r=json.loads((p/'complete.json').read_text())
            assert r['scores_sha256']==sha(p/'scores.npz')
    shutil.copy2(ROOT/'evaluation-cache/manifest.json',ROOT/'evaluation-cache-manifest.json')
    shutil.copy2(ROOT/'evaluation-cache/source_ids.npy',ROOT/'evaluation-source_ids.npy')
    files=['evaluation-launch-001.json','evaluation-progress.json','selection.json',
           'evaluation-cache-manifest.json','evaluation-source_ids.npy','models']
    files += [f'eval-{name}-{chip}' for name in names for chip in range(4)]
    archive=ROOT/'likelihood-study-001.tar.gz'
    with tarfile.open(archive,'w:gz') as f:
        for name in files:f.add(ROOT/name,arcname=name)
    idle_code='''import glob,json,subprocess,os
p=subprocess.run(['sudo','fuser']+glob.glob('/dev/vfio/[0-9]*'),capture_output=True,text=True)
print(json.dumps(dict(host=os.uname().nodename,vfio_returncode=p.returncode,vfio_owners=p.stdout.strip())))
'''
    def idle(i):return json.loads(execute(PEERS[i],'python3 -c '+shlex.quote(idle_code)))
    for attempt in range(12):
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:owners=list(pool.map(idle,range(4)))
        if all(p['vfio_returncode']==1 and not p['vfio_owners'] for p in owners):break
        time.sleep(5)
    else:raise RuntimeError('Scoring artifacts preserved, but chips remain occupied: '+json.dumps(owners))
    for i,p in enumerate(owners):assert p['host']==f't1v-n-51b3bdd2-w-{i}'
    receipt=dict(kind='R8_FROZEN_LIKELIHOOD_ARTIFACTS_NOT_OFFICIAL_OOD',
                 complete_shards=16,sha256=sha(archive),bytes=archive.stat().st_size,public_test_read=False,
                 final_owner_checks=owners,collector_source_sha256=sha(__file__))
    (ROOT/'likelihood-study-001.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
