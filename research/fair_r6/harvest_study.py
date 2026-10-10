"""Wait for this lab study, then gather completed small artifacts on its head."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import shlex
import shutil
import subprocess
import tarfile
import time
from launch_study import ROOT,PREVIOUS,PEERS,KEY,execute


def main():
    start=time.monotonic();previous=None
    code='''import pathlib,json,subprocess
r=pathlib.Path({root!r});host={host};rows=[]
for chip in range(4):
 seed=chip//2;shard=2*host+chip%2;unit=f'fair-r6-h{{host}}-c{{chip}}'
 path=r/f'field-s{{seed}}-{{shard}}/report.json'
 if path.exists():rows.append(dict(seed=seed,shard=shard,state='complete'))
 else:
  state=subprocess.run(['systemctl','show',unit,'-p','ActiveState','-p','Result'],capture_output=True,text=True).stdout
  if 'Result=failed' in state or 'ActiveState=failed' in state:
   log=subprocess.run(['journalctl','-u',unit,'-n','12','-o','cat','--no-pager'],capture_output=True,text=True).stdout
   rows.append(dict(seed=seed,shard=shard,state='failed',log=log))
  else:rows.append(dict(seed=seed,shard=shard,state='pending'))
print(json.dumps(rows))
'''
    def poll(host):
        return json.loads(execute(PEERS[host],'python3 -c '+shlex.quote(code.format(root=ROOT,host=host))))
    while True:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            groups=list(pool.map(poll,range(4)))
        rows=sum(groups,[]);complete=sum(x['state']=='complete' for x in rows)
        failures=[x for x in rows if x['state']=='failed']
        if failures:raise RuntimeError(json.dumps(failures))
        if complete!=previous:
            print(json.dumps(dict(stage='waiting',complete=complete,total=16,seconds=time.monotonic()-start)),flush=True)
            previous=complete
        if complete==16:break
        if time.monotonic()-start>900:raise TimeoutError('Study remains incomplete; preserve and inspect its units.')
        time.sleep(20)
    for host,ip in enumerate(PEERS[1:],start=1):
        for seed in range(2):
            for shard in range(2*host,2*host+2):
                subprocess.run(['scp','-q','-r','-F','/dev/null','-i',KEY,'-o','BatchMode=yes',
                                '-o','StrictHostKeyChecking=no',ip+':'+ROOT+f'/field-s{seed}-{shard}',ROOT+'/'],check=True)
    root=Path(ROOT)
    shutil.copy2(root/'normals/manifest.json',root/'normal-manifest.json')
    shutil.copy2(root/'normals/source_ids.npy',root/'normal-source_ids.npy')
    shutil.copy2(Path(PREVIOUS)/'closure-cache/manifest.json',root/'closure-manifest.json')
    shutil.copy2(Path(PREVIOUS)/'closure-cache/source_ids.npy',root/'closure-source_ids.npy')
    files=['launch-001.json','normal-manifest.json','normal-source_ids.npy','closure-manifest.json',
           'closure-source_ids.npy','pilot-s0','pilot-s1','refine-s0','refine-s1']
    for seed in range(2):
        for shard in range(8):
            name=f'field-s{seed}-{shard}';files.append(name)
            report=json.loads((root/name/'report.json').read_text())
            assert hashlib.sha256((root/name/'statistics.npz').read_bytes()).hexdigest()==report['output_sha256']
    archive=root/'study-001.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        for name in files:t.add(root/name,arcname=name)
    receipt=dict(stage='harvested',sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),
                 bytes=archive.stat().st_size,completed_shards=16)
    (root/'study-001.receipt.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt),flush=True)


if __name__=='__main__':main()
