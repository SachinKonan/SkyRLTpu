"""Rebuild normal PUCT admission exclusively from new CPU rewards."""
import hashlib,json
from pathlib import Path
from ttt_discover import State
from ttt_discover.tinker_utils.sampler import PUCTSampler
from tpu.science.training_env import PlacementTrainingEnv
from tpu.science.training_setup import check_references
from tpu.science.bootstrap import commit_rows,save,identity
from tpu.science.challenge_contract import CASES,aggregate
from tpu.science.feedback import observation,diagnostic_message
from tpu.science.seed_pool import verify_pool
from tpu.swarm.ray_train.config import Config
p=Path('.science/placement-cpu-20260917');targets=json.loads((p/'selected-seeds.json').read_text())
records={};references=[]
for host in range(8):
 folder=p/f'host-{host}'/'regrade'
 assert (folder/'complete.json').exists(),f'host {host} incomplete'
 for f in folder.glob('*.json'):
  r=json.loads(f.read_text())
  if 'source_id' in r:
   assert f.name not in records
   records[f.name]=r
  elif f.name.startswith('challenge_seed'):references.append(r)
check_references('placement',references,placement_backend='cpu')
report={}
for model,seeds in targets.items():
 results={}
 for i,seed in enumerate(seeds):
  cases=[records[f'{model}-{i:03d}-{case}.json'] for case in CASES]
  assert all(r['source_id']==seed['id'] for r in cases)
  results[hashlib.sha256(seed['code'].encode()).hexdigest()]=aggregate([r['result'] for r in cases])
 source=p/'source-seeds'/model if model!='muse' else Path('.science/placement-assessment-20260917/muse-final/bootstrap')
 rows=[json.loads(f.read_text()) for f in sorted(source.glob('layer-*/group-*/grade-*.json'))]
 roots={s.id:s for s in map(State.from_dict,json.loads((p/'source-seeds'/(model+'-roots.json')).read_text()))}
 for row in rows:
  if row['correctness']==1:
   r=results[hashlib.sha256(row['code'].encode()).hexdigest()]
   row.update(reward=r['reward'],correctness=r['correctness'],metrics=r['metrics'],feedback=observation('placement',r),message=diagnostic_message('placement',r))
 out=p/'cpu-seeds'/model;out.mkdir(parents=True,exist_ok=False)
 save(out/'puct_sampler_step_000000.json',dict(step=0,states=[s.to_dict() for s in roots.values()],initial_states=[s.to_dict() for s in roots.values()],puct_n={},puct_m={},puct_T=0))
 sampler=PUCTSampler(str(out/'puct_sampler.json'),env_type=PlacementTrainingEnv,problem_type='placement',batch_size=16,resume_step=0)
 commit_rows(sampler,roots,rows)
 pool=json.loads((out/'puct_sampler_step_000000.json').read_text());digest=identity(pool);verify_pool(out/'puct_sampler_step_000000.json',digest)
 retained=[s for s in pool['states'] if s.get('code')]
 assert len(retained)<=32
 report[model]=dict(unique_programs_regraded=len(seeds),unique_cpu_valid=sum(r['correctness']==1 for r in results.values()),retained=len(retained),maximum=32,best_reward=max(s['value'] for s in retained),pool_sha256=digest,optimizer_steps=0,grading=dict(backend='cpu',cpus=4,memory_gib=8,slots_per_host=16,cases=list(CASES)),admission='Top two per original root, then global code deduplication; CPU rewards only',source_scope='Previously valid completed grade records only; no ungraded generations or retries of originally invalid code')
 save(out/'seed-import.json',report[model])
 save(out/'cpu-candidate-results.json',results)
 profile=Path(f'tpu/swarm/ray_train/profiles/science-placement-v6e-{model}-cpu-seeded-train-001.json');config=json.loads(profile.read_text());config['seed_pool_sha256']=digest;Config.from_dict(config).validate();save(profile,config)
 print(model,report[model],flush=True)
save(p/'cpu-seed-summary.json',report)
