"""Read-only completion gate for the AC2 shared-best second round."""
import concurrent.futures,importlib.util,json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
s=importlib.util.spec_from_file_location('o',ROOT/'tpu/science/results/reallocation-10step-20260921/operations.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o);os.environ.update(o.ops.env)
from google.cloud import storage

def check(row,queue):
 c=storage.Client(project='vision-mix');b=c.bucket(row['bucket'][5:]);run=row['run_id'];prefix='ray-training/'+run+'/client/tinker_log/'+run+'/'
 matching=sorted([r for r in queue if r['run_id']==run],key=lambda r:r['job_id']);job=matching[-1] if matching else {}
 result=dict(run_id=run,model=row['model'],job_id=job.get('job_id'),status=job.get('status'),ready=False)
 blob=b.get_blob(prefix+'member_'+row['model']+'/checkpoints.jsonl')
 checkpoints=[json.loads(x) for x in blob.download_as_text(if_generation_match=blob.generation).splitlines() if x.strip()] if blob else []
 result['checkpoint']=max([x.get('batch',-1) for x in checkpoints],default=None)
 pool=b.get_blob(prefix+'puct_sampler_step_000010.json')
 result['final_pool']=dict(uri='gs://'+b.name+'/'+pool.name,generation=pool.generation) if pool else None
 result['ready']=job.get('status')=='SUCCEEDED' and any(x.get('batch')==10 for x in checkpoints) and pool is not None
 return result

def main():
 rows=[r for r in json.loads((o.HERE/'prepared.json').read_text())['jobs'] if r['kind']=='ac2'];assert len(rows)==3
 queue=o.queue()
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as e:checks=list(e.map(lambda r:check(r,queue),rows))
 result=dict(time=time.time(),ready=all(r['ready'] for r in checks),policy='all-three-models-start-from-best-completed-solution',runs=checks)
 path=HERE/'ac2-completion.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(result,indent=2)+'\n');tmp.replace(path);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
