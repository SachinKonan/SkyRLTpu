import hashlib,json,math
from pathlib import Path
from google.cloud import storage
from tpu.swarm.ray_train.config import Config
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
run='shu-v5p64-gemma-cp26-grpo-lr4e5-s1-20260920'
c=storage.Client(project='vision-mix');bucket=c.bucket('sk7524-tinker-tpu-us-central1')
source='math-v6e-gemma-cp26-grpo-lr4e5-s1-20260920';prefix='ray-training/'+source
summary=json.loads(bucket.blob(prefix+'/client/bootstrap/complete.json').download_as_bytes())
blob=bucket.get_blob(prefix+'/client/tinker_log/'+source+'/puct_sampler_step_000000.json');data=blob.download_as_bytes();pool=json.loads(data)
canonical=hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest()
assert canonical==summary['pool_sha256'] and pool['step']==0 and len(pool['states'])==51
assert all(s.get('code') and math.isfinite(s['value']) and s['value']>0 for s in pool['states'])
profile=json.loads((ROOT/'tpu/swarm/ray_train/profiles/fresh-v5p-gemma-cp26-grpo-lr4e5-s1-20260919.json').read_text())
profile.update(run_id=run,root='~/.cache/'+run,accelerator='tpu-v5p-64',hosts=8,zone='us-central1-a',bucket='gs://'+bucket.name,bootstrap_layers=0,bootstrap_all_hosts=False,bootstrap_max_drafts=0,bootstrap_target_valid=0,bootstrap_group_size=16,bootstrap_max_groups=32)
profile['client_env']['TTD_SICK_MARKER']='/home/gcpuser/.cache/'+run+'/runs/'+run+'/ENGINE-SICK'
profile['cache']['trainer_compile']='gs://'+bucket.name+'/'+run+'-trainer-compile-v1'
profile['cache']['inference_compile']='gs://'+bucket.name+'/'+run+'-inference-compile-v1'
path=ROOT/'tpu/swarm/ray_train/profiles'/f'{run}.json';path.write_text(json.dumps(profile,indent=2)+'\n')
cfg=Config.load(path);assert cfg.inference_hosts==7 and cfg.trainer.tp==4 and cfg.trainer.fsdp==1
seed=HERE/'seed-pool.json';seed.write_bytes(data)
record={'run_id':run,'profile':str(path.relative_to(ROOT)),'source_run_id':source,'source_generation':blob.generation,'source_pool_sha256':canonical,'file_sha256':hashlib.sha256(data).hexdigest(),'retained':len(pool['states']),'model_optimizer':'fresh; source step-1 checkpoint followed failed training','destination':cfg.run_gcs+'/client/tinker_log/'+run+'/puct_sampler_step_000000.json'}
(HERE/'seed-import.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
