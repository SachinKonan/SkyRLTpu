"""Prepare the authorized v6e qubit queue; no cloud writes or job submission."""
import argparse,hashlib,importlib.util,json,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
P=ROOT/'tpu/swarm/ray_train/profiles';OUT=ROOT/'.science/campaign-next-20260921'
def write(p,d):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(d,indent=2)+'\n')
def seeds():
 s=importlib.util.spec_from_file_location('o',ROOT/'tpu/science/results/reallocation-10step-20260921/operations.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o);os.environ.update(o.ops.env)
 from google.cloud import storage
 c=storage.Client(project='vision-mix');assert c._credentials.service_account_email=='289186856710-compute@developer.gserviceaccount.com'
 jobs=[]
 sources={'qwen':'science-v6e-qwen-qubit-grpo-lr15e4-s1-20260920','gemma':'science-v6e-gemma-qubit-grpo-lr4e5-s1-20260920','muse':'capacity-v6e-muse-qubit-grpo-4e5-20260920'}
 for model in ['gemma','qwen','muse']:
  source=sources[model];orig=json.loads((P/(source+'.json')).read_text())
  prefix='ray-training/'+source+'/client/tinker_log/'+source+'/'
  blob=c.bucket(orig['bucket'][5:]).get_blob(prefix+'puct_sampler_step_000000.json');assert blob
  data=blob.download_as_bytes(if_generation_match=blob.generation,timeout=180);pool=json.loads(data)
  assert pool['step']==0 and pool['states'] and all(s.get('code') and 0<s['value']<=1 for s in pool['states'])
  digest=hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest()
  run=f'next-v6e-{model}-qubit-10step-20260921';region='east5b' if model=='muse' else 'central1b';bucket='gs://sk7524-tinker-tpu-us-central1'
  base=json.loads((P/(f'science-v6e-{model}-qubit-grpo-lr'+('15e4' if model=='qwen' else '4e5')+'-s1-20260920.json')).read_text())
  base.update(run_id=run,root='~/.cache/'+run,bucket=bucket,zone='us-east5-b' if region=='east5b' else 'us-central1-b',bootstrap_layers=0,bootstrap_all_hosts=False,bootstrap_max_drafts=0,bootstrap_target_valid=0,seed_pool_sha256=digest,resume_min_checkpoint_step=0,checkpoint_resume=True,systemd_runtime=True,max_restarts_on_errors=3)
  base['client_env'].update(NUM_EPOCHS='10',TTD_SICK_MARKER=f'/home/gcpuser/.cache/{run}/runs/{run}/ENGINE-SICK')
  base['inference'].update(external_pool_updates=True,external_pool_lease_scope='run',external_pool_require_initial=False,external_pool_initial_wait_seconds=300,external_pool_scheduler=True,external_pool_attestation=True,external_pool_engines=4,external_pool_max_n=32,external_pool_max_concurrent_requests=4,external_pool_prepare_timeout=1800,external_pool_rpc_timeout=10,external_pool_lease_seconds=300,external_pool_heartbeat_seconds=30,external_pool_health_grace_seconds=90)
  for role in ['trainer','inference']:
   base['cache'][role+'_compile_seed']=orig['cache'][role+'_compile']
   base['cache'][role+'_compile']=bucket+'/campaign-next-20260921/'+run+'/'+role
  profile=P/(run+'.json');write(profile,base);folder=OUT/run;folder.mkdir(parents=True,exist_ok=True);(folder/'seed-pool.json').write_bytes(data)
  jobs.append(dict(run_id=run,model=model,kind='qubit',profile=str(profile.relative_to(ROOT)),pool='tpuswarm-v6e32-'+('east5b-qwen35' if region=='east5b' else 'central1b'),priority=100,bucket=bucket,epochs=10,source_run=source,source_uri='gs://'+blob.bucket.name+'/'+blob.name,source_generation=blob.generation,seed_file=str((folder/'seed-pool.json').relative_to(ROOT)),seed_file_sha256=hashlib.sha256(data).hexdigest(),seed_pool_sha256=digest,seed_count=len(pool['states']),seed_destination=bucket+'/ray-training/'+run+'/client/tinker_log/'+run+'/puct_sampler_step_000000.json',optimizer='fresh',adapter='fresh'))
 write(HERE/'prepared.json',dict(jobs=jobs));print('Prepared three profiles and pinned seed pools',flush=True)
def bundles():
 import yaml
 from tpu.swarm.ray_train.config import Config
 from tpu.swarm.ray_train.build import build
 doc=json.loads((HERE/'prepared.json').read_text())
 for r in doc['jobs']:
  cfg=Config.load(ROOT/r['profile']);assert cfg.client_env['NUM_EPOCHS']=='10' and cfg.accelerator=='tpu-v6e-32'
  archive,uri,task=build(ROOT/r['profile'],OUT/r['run_id']);d=yaml.safe_load(task.read_text());d['resources']['priority']=r['priority'];task.write_text(yaml.safe_dump(d,sort_keys=False))
  r.update(archive=str(archive.relative_to(ROOT)),archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest(),code_uri=uri,task=str(task.relative_to(ROOT)),task_sha256=hashlib.sha256(task.read_bytes()).hexdigest())
  print('Built',r['run_id'],flush=True)
 write(HERE/'prepared.json',doc)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['seeds','bundles']);a=p.parse_args();{'seeds':seeds,'bundles':bundles}[a.action]()
