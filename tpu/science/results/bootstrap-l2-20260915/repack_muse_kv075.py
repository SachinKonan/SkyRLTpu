"""Only add the 0.75 profile; preserve every byte of the previous code bundle."""
import copy,hashlib,io,json,tarfile
from pathlib import Path
import yaml
from tpu.swarm.ray_train.config import Config

for task in ('routing','placement'):
 old=f'science-{task}-v4-muse-bootstrap-l2-002';new=old[:-3]+'003'
 source=Path('.science')/('deployment-'+old)
 dest=Path('.science')/('deployment-'+new);dest.mkdir(exist_ok=True)
 manifest=json.loads((source/'manifest.json').read_text())
 archive=source/'science-training.tar.gz'
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==manifest['sha256']
 profile_name='tpu/swarm/ray_train/profiles/'+new+'.json'
 with tarfile.open(archive) as src:
  original=json.load(src.extractfile('tpu/swarm/ray_train/profiles/'+old+'.json'))
  profile=copy.deepcopy(original)
  profile['run_id']=new;profile['root']=profile['root'].replace(old,new)
  profile['client_env']['TTD_SICK_MARKER']=profile['client_env']['TTD_SICK_MARKER'].replace(old,new)
  profile['inference']['memory_utilization']=.75
  normalized=json.loads(json.dumps(profile).replace(new,old));normalized['inference']['memory_utilization']=.65
  assert normalized==original
  Config.from_dict(profile).validate()
  data=(json.dumps(profile,indent=2)+'\n').encode()
  assert not Path(profile_name).exists()
  Path(profile_name).write_bytes(data)
  with tarfile.open(dest/'science-training.tar.gz','w:gz') as out:
   for member in src:
    assert member.name!=profile_name
    out.addfile(member,src.extractfile(member) if member.isfile() else None)
   info=tarfile.TarInfo(profile_name);info.size=len(data);info.mode=0o644
   out.addfile(info,io.BytesIO(data))
 with tarfile.open(archive) as before,tarfile.open(dest/'science-training.tar.gz') as after:
  names={m.name for m in before};assert {m.name for m in after}==names|{profile_name}
  for member in before:
   if member.isfile():assert before.extractfile(member).read()==after.extractfile(member.name).read(),member.name
 digest=hashlib.sha256((dest/'science-training.tar.gz').read_bytes()).hexdigest()
 uri=profile['bucket']+'/code-bundles/science-training-'+digest+'.tar.gz'
 doc=yaml.safe_load((source/(old+'.yaml')).read_text())
 doc['name']=new;doc['run']=doc['run'].replace(old,new)
 doc['envs'].update(RAY_TRAIN_CODE=uri,RAY_TRAIN_CODE_SHA256=digest)
 yaml_path=dest/(new+'.yaml');yaml_path.write_text(yaml.safe_dump(doc,sort_keys=False))
 record=dict(manifest,run_id=new,sha256=digest,code_uri=uri,submitted=False,
   task_sha256=hashlib.sha256(yaml_path.read_bytes()).hexdigest(),
   parent_bundle_sha256=manifest['sha256'],original_bundle_members_byte_identical=True,
   profile_only_changes=['run_id','root','client_env.TTD_SICK_MARKER','inference.memory_utilization'],
   memory_utilization=.75)
 (dest/'manifest.json').write_text(json.dumps(record,indent=2)+'\n')
 print(json.dumps({'task':task,'profile':profile_name,'sha256':digest,'code_unchanged':True}),flush=True)
