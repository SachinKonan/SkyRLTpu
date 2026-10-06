import copy,hashlib,io,json,tarfile,tempfile,importlib.util
from pathlib import Path
import yaml
root=Path.cwd();base=root/'tpu/science/results/reallocation-10step-20260921';out=root/'.science/reallocation-10step/checkpoint-retry';out.mkdir(exist_ok=True)
original=(base/'prepared.json').read_bytes();manifest=json.loads(original);records=[]
helper=(root/'skyrl/utils/checkpoint_mirror.py').read_bytes()
for row in manifest['jobs']:
 if row['kind']!='circuit':continue
 previous=dict(row);folder=out/row['model'];folder.mkdir(exist_ok=True)
 source=root/row['archive'];assert hashlib.sha256(source.read_bytes()).hexdigest()==row['archive_sha256']
 archive=folder/'science-training.tar.gz';prefix='tpu/swarm/ray_train/source_overlay/';target=prefix+'skyrl/utils/checkpoint_mirror.py'
 with tarfile.open(source) as old:
  data={m.name:old.extractfile(m).read() for m in old if m.isfile()}
  assert target not in data
  overlay=data['tpu/swarm/ray_train/overlay.py'].decode()
  marker='DATABASE_FILES = {"skyrl/tinker/db_models.py"}'
  assert overlay.count(marker)==1
  overlay=overlay.replace(marker,marker+'\nCHECKPOINT_FILES = {"skyrl/utils/checkpoint_mirror.py"}')
  marker='        names.update(DATABASE_FILES)';assert overlay.count(marker)==1
  overlay=overlay.replace(marker,marker+'\n        names.update(CHECKPOINT_FILES)')
  marker='    allowed.extend([base | DATABASE_FILES for base in [set(), *allowed]])';assert overlay.count(marker)==1
  overlay=overlay.replace(marker,marker+'\n    allowed.extend([base | CHECKPOINT_FILES for base in [set(), *allowed]])')
  mf=json.loads(data[prefix+'manifest.json']);mf['skyrl/utils/checkpoint_mirror.py']=hashlib.sha256(helper).hexdigest()
  changes={'tpu/swarm/ray_train/overlay.py':overlay.encode(),prefix+'manifest.json':json.dumps(mf,sort_keys=True).encode(),target:helper}
  with tarfile.open(archive,'w:gz') as new:
   for member in old:
    info=copy.copy(member);b=changes.get(member.name,data.get(member.name));info.size=len(b) if b is not None else info.size
    new.addfile(info,io.BytesIO(b) if b is not None else None)
   info=tarfile.TarInfo(target);info.size=len(helper);info.mode=0o644;new.addfile(info,io.BytesIO(helper))
 with tarfile.open(archive) as new:
  actual={m.name:new.extractfile(m).read() for m in new if m.isfile()}
 assert set(actual)-set(data)=={target}
 assert {n for n in data if actual[n]!=data[n]}==set(changes)-{target}
 # Verify overlay installation and source identity with the packaged module.
 with tempfile.TemporaryDirectory() as td:
  td=Path(td);module=td/'overlay.py';module.write_text(overlay)
  spec=importlib.util.spec_from_file_location('candidate_overlay',module);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
  od=td/'source';od.mkdir()
  for n,b in actual.items():
   if n.startswith(prefix):
    p=od/n[len(prefix):];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b)
  assert mod.identity(od)!=hashlib.sha256(data[prefix+'manifest.json']).hexdigest()
  mod.install(od,td/'installed');assert (td/'installed/skyrl/utils/checkpoint_mirror.py').read_bytes()==helper
 sha=hashlib.sha256(archive.read_bytes()).hexdigest();uri=row['bucket']+'/code-bundles/science-training-'+sha+'.tar.gz'
 tasksrc=root/row['task'];assert hashlib.sha256(tasksrc.read_bytes()).hexdigest()==row['task_sha256']
 task=yaml.safe_load(tasksrc.read_text());task['envs'].update(RAY_TRAIN_CODE=uri,RAY_TRAIN_CODE_SHA256=sha)
 tp=folder/tasksrc.name;tp.write_text(yaml.safe_dump(task,sort_keys=False))
 row.update(archive=str(archive.relative_to(root)),archive_sha256=sha,code_uri=uri,task=str(tp.relative_to(root)),task_sha256=hashlib.sha256(tp.read_bytes()).hexdigest())
 records.append({'model':row['model'],'previous':previous,'replacement':dict(row),'changed_files':list(changes),'existing_files_verified':len(data)})
 print(row['model'],sha,flush=True)
(out/'prepared.json').write_text(json.dumps(manifest,indent=2)+'\n');(out/'build.json').write_text(json.dumps({'original_manifest_sha256':hashlib.sha256(original).hexdigest(),'bundles':records},indent=2)+'\n')
