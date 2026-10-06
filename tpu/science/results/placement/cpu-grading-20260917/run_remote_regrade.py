"""Run CPU-only reference checks and saved seeds through the real Ray v2 task."""
import hashlib,json,os,sys,time
from pathlib import Path
import ray
from tpu.science.placement_ray import grade_cpu_case
from tpu.science.challenge_contract import CASES
root=Path.cwd();rank=int(sys.argv[1]);out=root/'regrade';out.mkdir(exist_ok=True)
targets=json.loads((root/'selected-seeds.json').read_text())
ray.init(address='local',num_cpus=72,num_gpus=0,resources={'TPU':0,'placement_cpu_host':16},
         object_store_memory=1024**3,_memory=128*1024**3,include_dashboard=False,
         _temp_dir='/tmp/science-placement-cpu-regrade-'+str(os.getpid()),
         runtime_env={'env_vars':{'PYTHONPATH':str(root),'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'}})
refs={};report=[]
try:
 # Exercise both reference programs on every host before admitting saved programs.
 for name in ['challenge_seed.py','challenge_seed_jax.py']:
  for case in CASES:
   ref=grade_cpu_case.remote((root/'tpu/science'/name).read_text(),case,str(root),slots_per_host=16)
   refs[ref]=(name,case)
 while refs:
  ready,_=ray.wait(list(refs),num_returns=1,timeout=320)
  if not ready:raise TimeoutError('reference worker did not finish')
  for ref in ready:
   name,case=refs.pop(ref);r=ray.get(ref);report.append(r)
   (out/(name+'-'+case+'.json')).write_text(json.dumps(r,indent=2))
 if any(r['correctness']!=1 for r in report):raise RuntimeError('CPU reference failed')
 print(json.dumps(dict(event='references_passed',host_rank=rank,count=len(report))),flush=True)
 jobs=[]
 for model,seeds in targets.items():
  for i,seed in enumerate(seeds):
   for case in CASES:jobs.append((model,i,seed,case))
 for model,i,seed,case in jobs[rank::8]:
  key=f'{model}-{i:03d}-{case}';path=out/(key+'.json')
  if path.exists():continue
  refs[grade_cpu_case.remote(seed['code'],case,str(root),slots_per_host=16)]=(key,seed['id'])
 while refs:
  ready,_=ray.wait(list(refs),num_returns=1,timeout=3600)
  if not ready:raise TimeoutError('seed workers did not finish')
  for ref in ready:
   key,source_id=refs.pop(ref);r=ray.get(ref)
   (out/(key+'.json')).write_text(json.dumps(dict(source_id=source_id,result=r),indent=2))
   print(json.dumps(dict(event='seed_case_graded',host_rank=rank,key=key,valid=r['correctness'],reward=r['reward'],remaining=len(refs))),flush=True)
 (out/'complete.json').write_text(json.dumps(dict(rank=rank,ray=ray.__version__,time=time.time())))
finally:
 for ref in refs:ray.cancel(ref,force=True)
 ray.shutdown()
