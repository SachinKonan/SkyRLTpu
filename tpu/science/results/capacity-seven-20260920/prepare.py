"""Six additional model/problem runs, using verified original bootstrap pools."""
import hashlib,json,math
from pathlib import Path
import yaml
from google.cloud import storage
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.build import build
from tpu.science.package_training import package
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent;P=ROOT/'tpu/swarm/ray_train/profiles'
c=storage.Client(project='vision-mix')
plan=[('v5p','qwen','ac2','fresh-v4-qwen-ac2-grpo-lr15e4-s1-20260919','sk7524-tinker-tpu-us-central2'),('v5p','gemma','ac2','math-v6e-gemma-ac2-grpo-lr4e5-s1-20260920','sk7524-tinker-tpu-us-central1'),('v5p','gemma','qubit','science-v6e-gemma-qubit-grpo-lr4e5-s1-20260920','sk7524-tinker-tpu-us-east5'),('v6e','muse','ac2','math-v6e-muse-ac2-grpo-lr4e5-s1-20260920','sk7524-tinker-tpu-us-central1'),('v6e','qwen','qubit','science-v6e-qwen-qubit-grpo-lr15e4-s1-20260920','sk7524-tinker-tpu-us-east5'),('v6e','muse','qubit',None,None)]
rows=[]
for hw,model,task,source,src_bucket in plan:
 lr='15e4' if model=='qwen' else '4e5';run=f'capacity-{hw}-{model}-{task}-grpo-{lr}-20260920'
 folder=ROOT/'.science/packages/capacity-seven-20260920'/run
 assert not (folder/'submission.json').exists()
 if hw=='v5p':base=P/f'fresh-v5p-{model}-{task}-grpo-lr{lr}-s1-20260919.json'
 elif task=='ac2':base=P/f'math-v6e-{model}-ac2-grpo-lr{lr}-s1-20260920.json'
 else:base=P/f'science-v6e-{model}-qubit-grpo-lr{lr}-s1-20260920.json'
 d=json.loads(base.read_text());bucket='sk7524-tinker-tpu-us-east5' if hw=='v5p' else 'sk7524-tinker-tpu-us-central1'
 d.update(run_id=run,root='~/.cache/'+run,zone='us-east5-a' if hw=='v5p' else 'us-central1-b',bucket='gs://'+bucket)
 d['client_env']['TTD_SICK_MARKER']=f'/home/gcpuser/.cache/{run}/runs/{run}/ENGINE-SICK'
 for phase in ('trainer','inference'):d['cache'][phase+'_compile']=f'gs://{bucket}/{run}-{phase}-compile-v1'
 record=dict(model=model,task=task,hardware=hw,run_id=run,source_run_id=source,model_optimizer='fresh',priority=110 if task=='ac2' else 100,pool='tpuswarm-v5p32-east5a-erdos' if hw=='v5p' else 'tpuswarm-v6e32-central1b')
 if source:
  b=c.bucket(src_bucket);prefix='ray-training/'+source
  completed=json.loads(b.blob(prefix+'/client/bootstrap/complete.json').download_as_bytes())
  blob=b.get_blob(prefix+'/client/tinker_log/'+source+'/puct_sampler_step_000000.json');data=blob.download_as_bytes();pool=json.loads(data)
  canonical=hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest()
  assert canonical==completed['pool_sha256'] and pool['step']==0 and pool['states']
  assert all(s.get('code') and math.isfinite(s['value']) and s['value']>0 for s in pool['states'])
  if task=='qubit':assert all(s['value']<=1 for s in pool['states'])
  d.update(bootstrap_layers=0,bootstrap_all_hosts=False,bootstrap_max_drafts=0,bootstrap_target_valid=0)
  if task=='qubit':d['seed_pool_sha256']=canonical
  folder.mkdir(parents=True,exist_ok=True);(folder/'seed-pool.json').write_bytes(data)
  record.update(seed_file=str((folder/'seed-pool.json').relative_to(ROOT)),seed_file_sha256=hashlib.sha256(data).hexdigest(),seed_pool_sha256=canonical,source_generation=blob.generation,retained=len(pool['states']),seed_destination=f'gs://{bucket}/ray-training/{run}/client/tinker_log/{run}/puct_sampler_step_000000.json')
 else:record['retained']=0;record['bootstrap']='fresh bounded bootstrap; 1024 draft cap, 512 distinct valid target'
 profile=P/(run+'.json');profile.write_text(json.dumps(d,indent=2)+'\n');cfg=Config.load(profile)
 if task=='qubit':
  task_yaml=package(profile,folder);archive=folder/'science-training.tar.gz'
 else:
  archive,_,task_yaml=build(profile,folder)
  doc=yaml.safe_load(task_yaml.read_text());audit=(ROOT/'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text();doc['run']="set -euo pipefail\npython3 - <<'CLEAN_HOST'\n"+audit+'\nCLEAN_HOST\n'+doc['run'];task_yaml.write_text(yaml.safe_dump(doc,sort_keys=False))
 doc=yaml.safe_load(task_yaml.read_text());doc['resources']['priority']=record['priority'];task_yaml.write_text(yaml.safe_dump(doc,sort_keys=False))
 record.update(profile=str(profile.relative_to(ROOT)),package_dir=str(folder.relative_to(ROOT)),archive=str(archive.relative_to(ROOT)),task_yaml=str(task_yaml.relative_to(ROOT)),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),yaml_sha256=hashlib.sha256(task_yaml.read_bytes()).hexdigest(),code_uri=doc['envs']['RAY_TRAIN_CODE'])
 if task=='qubit':
  m=json.loads((folder/'manifest.json').read_text());m['task_sha256']=record['yaml_sha256'];(folder/'manifest.json').write_text(json.dumps(m,indent=2)+'\n')
 (folder/'submission-manifest.json').write_text(json.dumps(record,indent=2)+'\n');rows.append(record);print(model,task,hw,'seeds',record['retained'],flush=True)
(HERE/'jobs.json').write_text(json.dumps({'jobs':rows},indent=2)+'\n')
