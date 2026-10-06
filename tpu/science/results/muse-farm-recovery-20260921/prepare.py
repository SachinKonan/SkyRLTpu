import json,tarfile,io,hashlib,copy,yaml
from pathlib import Path
root=Path.cwd();d=root/'.science/muse-farm-memory-fix-20260921'
r=next(r for r in json.loads((root/'tpu/science/results/reallocation-10step-20260921/prepared.json').read_text())['jobs'] if r['run_id']=='farm10-muse-1-20260921')
src=root/r['archive'];assert hashlib.sha256(src.read_bytes()).hexdigest()==r['archive_sha256']
archive=d/'ray-training.tar.gz';changed=[]
with tarfile.open(src,'r:gz') as old,tarfile.open(archive,'w:gz') as new:
 for m in old:
  data=old.extractfile(m).read() if m.isfile() else None
  if m.name==r['profile']:
   a=json.loads(data);b=copy.deepcopy(a);assert a['inference']['memory_utilization']==0.8;b['inference']['memory_utilization']=0.7
   data=(json.dumps(b,indent=2)+'\n').encode();m=copy.copy(m);m.size=len(data);changed.append(m.name)
  new.addfile(m,io.BytesIO(data) if data is not None else None)
assert changed==[r['profile']]
# Verify every other archive entry, including metadata and links, is identical.
with tarfile.open(src) as a,tarfile.open(archive) as b:
 aa={m.name:m for m in a};bb={m.name:m for m in b};assert aa.keys()==bb.keys()
 for n,m in aa.items():
  z=bb[n];assert (m.mode,m.type,m.linkname,m.uid,m.gid,m.mtime)==(z.mode,z.type,z.linkname,z.uid,z.gid,z.mtime)
  if n!=r['profile']:
   assert m.size==z.size
   if m.isfile():assert a.extractfile(m).read()==b.extractfile(z).read(),n
  else:
   x=json.load(a.extractfile(m));y=json.load(b.extractfile(z));y['inference']['memory_utilization']=0.8;assert x==y
sha=hashlib.sha256(archive.read_bytes()).hexdigest();uri=r['bucket']+'/code-bundles/ray-training-'+sha+'.tar.gz'
a=yaml.safe_load((root/r['task']).read_text());b=copy.deepcopy(a);b['envs']['RAY_TRAIN_CODE']=uri;b['envs']['RAY_TRAIN_CODE_SHA256']=sha
check=copy.deepcopy(b);check['envs']=a['envs'];assert check==a
p=d/'farm10-muse-1-20260921.yaml';p.write_text(yaml.safe_dump(b,sort_keys=False))
manifest={**r,'old_job_id':1366,'previous_job_id':1333,'original_archive_sha256':r['archive_sha256'],'archive':str(archive),'archive_sha256':sha,'task':str(p),'task_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'code_uri':uri,'change':{'profile':r['profile'],'field':'inference.memory_utilization','before':0.8,'after':0.7}}
(d/'prepared.json').write_text(json.dumps(manifest,indent=2)+'\n');print(json.dumps({'sha256':sha,'change':manifest['change'],'entries_checked':len(aa)}))
