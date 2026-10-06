import importlib.util,json,os,time,copy,hashlib,yaml
from pathlib import Path
s=importlib.util.spec_from_file_location('o','.science/reallocation-10step/ops.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o);os.environ.update(o.ops.env)
from google.cloud import storage
out=o.ROOT/'tpu/science/results/qwen-ac2-central-20260921';out.mkdir(exist_ok=True)
queue=o.queue();job=next(r for r in queue if r['job_id']==1334);assert job['status'] in ['PENDING','RECOVERING'],job
pools=o.sky('jobs','pool','status','-a');idle=[l for l in pools.splitlines() if l.startswith('tpuswarm-v6e32-central1b') and 'READY' in l and l.split()[-1]=='-'];assert idle
worker='tpuswarm-v6e32-central1b-'+idle[0].split()[1]
audit=o.ssh(worker,(o.ROOT/'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text())
identity=o.run(['gcloud','auth','list','--filter=status:ACTIVE','--format=value(account)']).strip();c=storage.Client(project='vision-mix');assert identity==c._credentials.service_account_email=='289186856710-compute@developer.gserviceaccount.com'
row=next(r for r in json.loads((o.HERE/'prepared.json').read_text())['jobs'] if r['kind']=='ac2' and r['model']=='qwen');assert row['run_id']==job['run_id']
archive=o.ROOT/row['archive'];assert hashlib.sha256(archive.read_bytes()).hexdigest()==row['archive_sha256']
bn,n=row['code_uri'][5:].split('/',1);blob=c.bucket(bn).get_blob(n);assert blob and hashlib.sha256(blob.download_as_bytes(if_generation_match=blob.generation)).hexdigest()==row['archive_sha256']
b=c.bucket(row['bucket'][5:]);prefix='ray-training/'+row['run_id'];index=b.get_blob(prefix+'/client/tinker_log/'+row['run_id']+'/member_qwen/checkpoints.jsonl');rows=[json.loads(l) for l in index.download_as_text(if_generation_match=index.generation).splitlines() if l.strip()];cp=rows[-1];assert cp['batch']>=3
model=cp['state_path'].split('/')[2];objects=[]
for name in [prefix+'/client/tinker_log/'+row['run_id']+'/puct_sampler_step_'+str(cp['batch']).zfill(6)+'.json',prefix+'/tinker-backup.db',prefix+'/checkpoints/'+model+'/'+cp['name']+'.tar.gz',prefix+'/checkpoints/'+model+'/sampler_weights/'+cp['name']+'.tar.gz']:
 v=b.get_blob(name);assert v and v.size>0;objects.append({'name':name,'generation':v.generation,'size':v.size})
taskpath=o.ROOT/row['task'];assert hashlib.sha256(taskpath.read_bytes()).hexdigest()==row['task_sha256'];old=yaml.safe_load(taskpath.read_text());task=copy.deepcopy(old);assert task['resources']['zone']=='us-east5-b';task['resources']['zone']='us-central1-b';check=copy.deepcopy(task);check['resources']['zone']='us-east5-b';assert check==old
p=out/(row['run_id']+'.yaml');p.write_text(yaml.safe_dump(task,sort_keys=False))
result={'model':'qwen','run_id':row['run_id'],'old_job_id':1334,'old_cluster':job['cluster'],'pool':'tpuswarm-v6e32-central1b','task':str(p.relative_to(o.ROOT)),'task_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_task_sha256':row['task_sha256'],'code_uri':row['code_uri'],'code_sha256':row['archive_sha256'],'checkpoint':cp,'checkpoint_index_generation':index.generation,'objects':objects,'change':{'resources.zone':['us-east5-b','us-central1-b']}}
(out/'prepared.json').write_text(json.dumps({'time':time.time(),'jobs':[result]},indent=2)+'\n');(out/'preflight.json').write_text(json.dumps({'job':job,'idle':idle,'audited_worker':worker,'audit':json.loads(audit),'identity':identity},indent=2)+'\n')
print(json.dumps({'old_job':job,'saved_step':cp['batch'],'central_idle':len(idle),'audited_worker':worker,'task':str(p)}))
