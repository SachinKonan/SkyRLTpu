import importlib.util,json,subprocess,os
from pathlib import Path
s=importlib.util.spec_from_file_location('ops_env','/scratch/gpfs/ZHUANGL/sk7524/.cache/native-multilora-v432-20260919/ops_env.py')
ops=importlib.util.module_from_spec(s);s.loader.exec_module(ops)
ROOT=Path('/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-hybrid-inference-migration')
HERE=ROOT/'tpu/science/results/reallocation-10step-20260921'
OUT=ROOT/'.science/reallocation-10step'
def run(args,timeout=120,input=None):
 p=subprocess.run([str(a) for a in args],env=ops.env,capture_output=True,text=True,timeout=timeout,input=input)
 if p.returncode:raise RuntimeError(f'{args[0]} exit {p.returncode}: '+p.stderr[-1000:]+p.stdout[-1000:])
 return p.stdout

def sky(*args):return run([ops.venv/'sky',*args],timeout=300)
def queue():
 code="""import sky,json
rows,*_=sky.get(sky.jobs.queue_v2(refresh=True,skip_finished=False,fields=['job_id','job_name','status','current_cluster_name','pool','priority']))
print(json.dumps([dict(job_id=r.job_id,run_id=r.job_name,status=getattr(r.status,'value',str(r.status)),cluster=r.current_cluster_name,pool=r.pool,priority=r.priority or 0) for r in rows]))
"""
 return json.loads(run([ops.venv/'python','-c',code]))
def ssh(cluster,code,timeout=60):
 f=Path(ops.env['HOME'])/'.sky/generated/ssh'/cluster
 return run(['ssh','-F',f,'-o','BatchMode=yes','-o','ConnectTimeout=8',cluster,'python3 -'],timeout=timeout,input=code)
if __name__=='__main__':
 import concurrent.futures
 jobs={'queue':lambda:json.dumps(queue(),indent=2),'pools':lambda:sky('jobs','pool','status','-a'),
 'identity':lambda:run(['gcloud','auth','list','--filter=status:ACTIVE','--format=value(account)'])}
 for zone in ['us-east5-a','us-east5-b','us-central1-a','us-central1-b','us-central2-b']:
  for kind in ['tpu-vm','queued-resources']:
   jobs[zone+'-'+kind]=lambda z=zone,k=kind:run(['gcloud','compute','tpus',k,'list','--project','vision-mix','--zone',z,'--format=json'])
 def work(pair):
  key,fn=pair
  try:
   data=fn();(OUT/(key+'.json' if key!='pools' else key+'.txt')).write_text(data);print(key,'ok',flush=True)
  except Exception as e:print(key,str(e),flush=True)
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as e:list(e.map(work,jobs.items()))
