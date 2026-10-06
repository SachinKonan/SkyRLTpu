import importlib.util,json,hashlib,base64,re,time,fcntl,os,sqlite3
from pathlib import Path
s=importlib.util.spec_from_file_location('o','.science/reallocation-10step/ops.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o);os.environ.update(o.ops.env)
from google.cloud import storage
d=o.ROOT/'.science/muse-farm-memory06-20260921';lock=(d/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(n,x):
 p=d/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
assert not (d/'intent.json').exists(),'reconcile existing intent'
r=json.loads((d/'prepared.json').read_text());archive=Path(r['archive']);assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['archive_sha256'];assert hashlib.sha256(Path(r['task']).read_bytes()).hexdigest()==r['task_sha256']
c=storage.Client(project='vision-mix');assert o.run(['gcloud','auth','list','--filter=status:ACTIVE','--format=value(account)']).strip()==c._credentials.service_account_email=='289186856710-compute@developer.gserviceaccount.com'
bn,n=r['code_uri'][5:].split('/',1);blob=c.bucket(bn).blob(n)
if not blob.exists():blob.upload_from_filename(archive,if_generation_match=0,timeout=120)
blob.reload();assert blob.size==archive.stat().st_size and blob.md5_hash==base64.b64encode(hashlib.md5(archive.read_bytes()).digest()).decode();save('upload.json',{'uri':r['code_uri'],'generation':blob.generation,'size':blob.size,'sha256':r['archive_sha256'],'md5_verified':True})
q=o.queue();a=next(x for x in q if x['job_id']==1368);assert a['run_id']==r['run_id'] and a['status']=='PENDING' and a['cluster']=='tpuswarm-v4-32-central2-smoke-177',a
query="from sky.skylet import job_lib;import json;rs=job_lib.load_job_queue(job_lib.dump_job_queue(None,True));print(json.dumps([{k:r.get(k) for k in ['job_id','status']} for r in rs],default=str))"
remote='import subprocess\np=subprocess.run(["/home/gcpuser/skypilot-runtime/bin/python","-c",'+repr(query)+'],capture_output=True,text=True,timeout=30)\nassert p.returncode==0,p.stderr\nprint(p.stdout)'
rows=json.loads(o.ssh(a['cluster'],remote));assert all(x['status'] in ['JobStatus.FAILED','JobStatus.CANCELLED','JobStatus.SUCCEEDED'] for x in rows),rows
con=sqlite3.connect('file:'+str(Path(o.ops.env['HOME'])/'.sky/api_server/requests.db')+'?mode=ro',uri=True)
active=con.execute("select request_id,status from requests where status in ('PENDING','WAITING','RUNNING') and cluster_name=? and name='sky.exec'",(a['cluster'],)).fetchall();assert not active,active
save('preflight.json',{'time':time.time(),'jobs':[x for x in q if x['pool']==r['pool'] and x['status'] not in ['CANCELLED','SUCCEEDED','FAILED']],'old_remote_jobs':rows,'active_exec_requests':active})
save('intent.json',{'time':time.time(),'state':'cancelling','old_job_id':1368})
(d/'cancel.txt').write_text(o.sky('jobs','cancel','1368','--yes'))
for _ in range(60):
 q=o.queue();a=next(x for x in q if x['job_id']==1368)
 if a['status']=='CANCELLED':break
 time.sleep(3)
else:raise RuntimeError('cancel unconfirmed')
assert not any(x['run_id']==r['run_id'] and x['status'] in {'RUNNING','STARTING','PENDING','RECOVERING','SUBMITTED','CANCELLING'} for x in q)
save('cancelled.json',a);save('intent.json',{'time':time.time(),'state':'submitting','old_job_id':1368})
out=o.sky('jobs','launch',r['task'],'--pool',r['pool'],'--yes','--detach-run');(d/'submission.txt').write_text(out)
ids=re.findall(r'(?:Job ID|Job id|job ID|job id)[:\s]+(\d+)',out);assert ids,'uncertain launch; reconcile'
id=int(ids[-1]);save('intent.json',{'time':time.time(),'state':'submitted','old_job_id':1368,'job_id':id,'run_id':r['run_id']})
q=o.queue();new=next(x for x in q if x['job_id']==id);assert new['run_id']==r['run_id'];save('queue-after.json',[x for x in q if x['job_id'] in [1368,id,1331,1371]]);print(json.dumps(new))
