"""Resume the three authorized qubit runs to 25 after durable source completion.

Usage: python -m tpu.science.ops.qubit_continue25 --once|--watch
Only the manifests under extend25 are eligible; no pools are created or resized.
"""
from tpu.science.ops.routing_relaunch_watch import Rollout, BASE, POOL, save
from pathlib import Path
import argparse, fcntl, hashlib, json, re, time, os, socket

def resume_object_keys(checkpoint):
 """Validate the indexed state archive; model IDs can change after recovery."""
 match=re.fullmatch(r'tinker://([A-Za-z0-9_-]+)/weights/([A-Za-z0-9_-]+)',checkpoint['state_path'])
 if match is None:raise ValueError('Invalid checkpoint state_path')
 model_id,name=match.groups()
 # A recovery after the last update can create only `final` under a new model
 # ID. The numbered checkpoint still belongs to the previous incarnation.
 return ['tinker-backup.db',f'checkpoints/{model_id}/{name}.tar.gz']

def require_checkpoint_registry(database, checkpoint):
 """An archive alone cannot resume: the restored API registry must know it."""
 import sqlite3
 key=resume_object_keys(checkpoint)[1]
 _,model_id,archive=key.split('/')
 name=archive.removesuffix('.tar.gz')
 with sqlite3.connect(Path(database).resolve().as_uri()+'?mode=ro',uri=True) as db:
  model=db.execute('SELECT base_model FROM models WHERE model_id=?',(model_id,)).fetchone()
  row=db.execute('SELECT status FROM checkpoints WHERE model_id=? AND checkpoint_id=? '
                 'AND checkpoint_type=?',(model_id,name,'TRAINING')).fetchone()
  if not model or not row or row[0]!='COMPLETED':
   raise RuntimeError(f'Durable database cannot resume completed checkpoint {model_id}/{name}: {row}')
 return {'model_id':model_id,'checkpoint_id':name,'base_model':model[0]}

def source_ready(source, record):
 """Permit an audited shutdown failure only after its old job is terminal.

 The checkpoint, metrics, search state and archive checks in submit still run.
 A saved checkpoint alone must never authorize two writers to the same run.
 """
 if source['status']=='SUCCEEDED':return True
 if source['status'] in ('RUNNING','RECOVERING','STARTING','PENDING','CANCELLING'):return False
 proof=record.get('shutdown_recovery')
 if source['status'] in ('CANCELLED','FAILED','FAILED_DRIVER') and proof:
  assert proof['source_job']==source['job_id']==record['source_job']
  assert proof['run_id']==source['name']==record['run_id']
  assert proof['completed_step']==record['minimum_resume_step']
  assert proof['client_exit_code']==0 and proof['failure_phase']=='final_run_writeback'
  evidence=Path(proof['evidence'])
  assert hashlib.sha256(evidence.read_bytes()).hexdigest()==proof['evidence_sha256']
  return True
 raise RuntimeError('Source did not complete successfully: '+str(source))

def submit(record):
 jid=record['source_job'];folder=base/str(jid)
 with (folder/'submit.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX)
  if (folder/'submitted.json').exists():return 'submitted'
  if (folder/'intent.json').exists():raise RuntimeError('Ambiguous prior launch; manual reconciliation required')
  source=r.job(jid)
  assert source['name']==record['run_id'] and source['pool']==POOL
  if not source_ready(source,record):return 'waiting_source'
  active=r.query("i.name=? AND s.status IN ('RUNNING','RECOVERING','STARTING','PENDING')",(record['run_id'],))
  if active:raise RuntimeError('Another job owns this run identity: '+str(active))
  for key in ('profile','task'):
   assert hashlib.sha256(Path(record[key]).read_bytes()).hexdigest()==record[key+'_sha256'], 'Prepared file changed: '+key
  c=json.loads(Path(record['profile']).read_text());bucket=c['bucket'].removeprefix('gs://');prefix='ray-training/'+c['run_id']
  assert c['run_id']==record['run_id'] and c['client_env']['NUM_EPOCHS']=='25'
  assert c['checkpoint_resume'] and c['client_env']['TTD_RESUME_STRICT']=='1'
  assert c['resume_min_checkpoint_step']==record['minimum_resume_step']
  def read(key):return r.storage.bucket(bucket).blob(prefix+'/'+key).download_as_text()
  member=c['client_env']['TTD_ANSWER_MODEL_FAMILY'];member='qwen' if member.startswith('qwen') else member
  # Use the member tag from the model preset for Qwen families.
  if 'qwen' in c['model_preset']:member='qwen'
  if 'gemma' in c['model_preset']:member='gemma'
  index=[json.loads(l) for l in read('client/tinker_log/'+c['run_id']+'/member_'+member+'/checkpoints.jsonl').splitlines() if l.strip()]
  last=index[-1];assert last['batch']==record['minimum_resume_step'] and last.get('state_path'),last
  state=json.loads(read('client/tinker_log/'+c['run_id']+f"/puct_sampler_step_{record['minimum_resume_step']:06d}.json"));assert state['states']
  metrics=json.loads(read('client/tinker_log/'+c['run_id']+'/metrics.jsonl').splitlines()[-1]);assert metrics['step']==record['minimum_resume_step']
  blobs=[]
  for key in resume_object_keys(last):
   b=r.storage.bucket(bucket).blob(prefix+'/'+key);b.reload();assert b.size>0;blobs.append({'key':key,'size':b.size,'generation':b.generation})
   if key=='tinker-backup.db':
    snapshot=folder/'resume-registry.db'
    b.download_to_filename(str(snapshot),if_generation_match=b.generation)
    require_checkpoint_registry(snapshot,last)
  nodes=json.loads(r.ops.run(['gcloud','compute','tpus','tpu-vm','list','--zone','us-central2-b','--project','vision-mix','--format=json']))
  healthy=[n['name'] for n in nodes if n.get('acceleratorType')=='v4-64' and n.get('state')=='READY' and n.get('health')=='HEALTHY'];assert healthy
  assert hashlib.sha256(Path(record['archive']).read_bytes()).hexdigest()==record['code_sha256']
  r.upload(record['archive'],record['code_uri'])
  intent={'time':time.time(),'source':source,'record':record,'resume_checkpoint':last,'durable':blobs,'pool_jobs':r.query("i.pool=? AND s.status IN ('RUNNING','RECOVERING','STARTING','PENDING')",(POOL,)),'healthy_v4_nodes':healthy}
  save(folder/'intent.json',intent)
  raw=r.ops.sky('jobs','launch',record['task'],'--pool',POOL,'--yes','--detach-run')
  (folder/'launch.txt').write_text(raw)
  ids=re.findall(r'(?:Job ID|Job id|job ID|job id)[:\s]+(\d+)',raw)
  if not ids:raise RuntimeError('Launch receipt ambiguous; do not retry')
  new_id=int(ids[-1]);new=r.job(new_id);assert new['name']==record['run_id'] and new['pool']==POOL
  save(folder/'submitted.json',{'time':time.time(),'source_job':jid,'job':new,'target_steps':record['target_steps']})
  print(json.dumps({'event':'submitted','source':jid,'job':new}),flush=True);return 'submitted'

def dispatch_request(record, request_id):
 """Execute only this continuation's existing, still-pending Sky request."""
 import yaml
 from sky.server.requests import requests, executor
 receipt=json.loads((base/str(record['source_job'])/'submitted.json').read_text())
 job=r.job(receipt['job']['job_id'])
 req=requests.get_request(request_id)
 assert req.name=='sky.exec' and req.cluster_name==job['cluster']
 tasks=[d for d in yaml.safe_load_all(req.request_body.task) if isinstance(d,dict) and d.get('name')]
 assert len(tasks)==1
 task=tasks[0];env=task.get('envs',{})
 assert task['name']==job['name']==record['run_id']
 assert str(env.get('SKYPILOT_MANAGED_JOB_ID'))==str(job['job_id'])
 assert env.get('RAY_TRAIN_CODE_SHA256')==record['code_sha256']
 assert job['pool']==record.get('pool',POOL)
 if job['status'] not in ('STARTING','RECOVERING') or req.status!=requests.RequestStatus.PENDING:return
 executor.executor_initializer('qubit-continue25-'+str(job['job_id']))
 executor._request_execution_wrapper(request_id,False,num_db_connections_per_worker=1)
 print(json.dumps({'request':request_id,'status':requests.get_request(request_id,fields=['status']).status.value}),flush=True)


def check_start(record):
 """Repair the known pending-exec issue without submitting another job."""
 import sqlite3,subprocess,sys
 folder=base/str(record['source_job'])
 receipt=json.loads((folder/'submitted.json').read_text());job=r.job(receipt['job']['job_id'])
 if job['status'] in ('RUNNING','SUCCEEDED'):return {'status':'started','job':job}
 if job['status'] not in ('PENDING','STARTING','RECOVERING'):raise RuntimeError('Continuation stopped: '+str(job))
 if job['status'] not in ('STARTING','RECOVERING') or not job['cluster']:return {'status':'submitted','job':job}
 db=Path(r.ops.ops.env['HOME'])/'.sky/api_server/requests.db'
 with sqlite3.connect('file:'+str(db)+'?mode=ro',uri=True) as conn:
  # Only small metadata; do not load or log credential-bearing request bodies.
  rows=conn.execute("SELECT request_id,name,status,created_at,cluster_name FROM (SELECT request_id,name,status,created_at,cluster_name FROM requests ORDER BY rowid DESC LIMIT 1000) WHERE name='sky.exec' AND status='PENDING' AND cluster_name=?",(job['cluster'],)).fetchall()
 for rid,_,_,created,_ in rows:
  if created<receipt['time']-120 or time.time()-created<120:continue
  previous=children.get(rid)
  if previous:continue  # One repair attempt per request; never repeatedly dispatch it.
  log=(folder/('dispatch-'+rid+'.log')).open('a')
  child=subprocess.Popen([sys.executable,'-m','tpu.science.ops.qubit_continue25','--dispatch',rid,'--source',str(record['source_job'])],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  log.close();children[rid]=child
  save(folder/'dispatch.json',{'time':time.time(),'request':rid,'pid':child.pid,'job':job})
 return {'status':'submitted','job':job}


def main():
 global r,base,children
 parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
 mode.add_argument('--once',action='store_true');mode.add_argument('--watch',action='store_true');mode.add_argument('--dispatch')
 parser.add_argument('--source',type=int);args=parser.parse_args()
 base=BASE/'extend25';records=json.loads((base/'prepared.json').read_text());children={}
 minimum_steps={1493:10,1590:15,1600:15}
 assert {x['source_job'] for x in records}==set(minimum_steps)
 assert all(minimum_steps[x['source_job']]<=x['minimum_resume_step']<x['target_steps'] for x in records)
 assert all(x['target_steps']==25 for x in records)
 r=Rollout()
 if args.dispatch:
  dispatch_request(next(x for x in records if x['source_job']==args.source),args.dispatch);return
 lock=(base/'watch.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 # Hold the old watcher's lock too, preventing accidental double scheduling.
 oldlock=(BASE/'extend15/gemma-watch.lock').open('a');fcntl.flock(oldlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert (BASE/'extend15/superseded-by-extend25.json').exists()
 assert not (BASE/'extend15/1493/intent.json').exists()
 save(base/'watch-process.json',{'time':time.time(),'pid':os.getpid(),'host':socket.gethostname(),'module':__name__,'target_steps':25})
 deadline=time.time()+7*86400;blocked={}
 while time.time()<deadline:
  statuses=[]
  for record in records:
   jid=record['source_job'];folder=base/str(jid)
   if jid in blocked:
    statuses.append(blocked[jid]);continue
   try:
    status=submit(record)
    d={'source_job':jid,'source':r.job(jid),'status':status,'target_steps':25}
    if status=='submitted':d.update(check_start(record))
   except Exception as exc:
    fatal=isinstance(exc,(AssertionError,RuntimeError)) or (folder/'intent.json').exists()
    d={'source_job':jid,'status':'blocked' if fatal else 'retrying_read','error':str(exc)[:1500],'target_steps':25}
    save(folder/'error.json',dict(d,time=time.time()))
    if fatal:blocked[jid]=d
   statuses.append(d)
  save(base/'watch-status.json',{'time':time.time(),'runs':statuses})
  print(json.dumps({'time':time.time(),'runs':statuses}),flush=True)
  if args.once or all(d['status'] in ('started','blocked') for d in statuses):break
  time.sleep(60)
 else:raise RuntimeError('Step-25 watcher expired after seven days')
 if blocked:raise RuntimeError('Some continuations need manual reconciliation; see watch-status.json')

if __name__=='__main__':main()
