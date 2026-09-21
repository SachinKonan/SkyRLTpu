import importlib.util,json,hashlib,os,fcntl,subprocess,time
from pathlib import Path
s=importlib.util.spec_from_file_location('o','.science/reallocation-10step/ops.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o);os.environ.update(o.ops.env)
from google.cloud import storage
OUT=o.OUT/'checkpoint-retry';build=json.loads((OUT/'build.json').read_text())
def save(p,x):
 tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(x,indent=2)+'\n');tmp.replace(p)
with (o.OUT/'deploy.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 repair=json.loads((o.OUT/'checkpoint-registration-repair.json').read_text());assert repair['state']=='published' and repair['step']==1
 assert len(repair['verified_archives'])==2
 c=storage.Client(project='vision-mix');assert c._credentials.service_account_email=='289186856710-compute@developer.gserviceaccount.com'
 assert o.run(['gcloud','auth','list','--filter=status:ACTIVE','--format=value(account)']).strip()==c._credentials.service_account_email
 machine=json.loads(o.run(['gcloud','compute','tpus','tpu-vm','describe','shu-v5p-64-on-demand-machine1','--zone','us-central1-a','--project','vision-mix','--format=json']))
 assert machine['state']=='READY' and machine['acceleratorType']=='v5p-64'
 state=json.loads((o.HERE/'circuit-queue-state.json').read_text());assert state['phase']=='blocked' and state['attempt']==5 and state['index']==0
 assert state['external_ips']==[e['accessConfig']['externalIp'] for e in machine['networkEndpoints']]
 row=next(r for r in json.loads((o.HERE/'prepared.json').read_text())['jobs'] if r['kind']=='circuit' and r['model']=='gemma')
 b=c.bucket(row['bucket'][5:]);prefix='ray-training/'+row['run_id']
 cp=json.loads(b.blob(prefix+'/client/tinker_log/'+row['run_id']+'/member_gemma/checkpoints.jsonl').download_as_text().strip());assert cp==repair['checkpoint_index']
 for a in repair['verified_archives']:
  bn,n=a['uri'][5:].split('/',1);blob=c.bucket(bn).get_blob(n);assert str(blob.generation)==str(a['generation']) and blob.md5_hash==a['md5_hash']
 receipts=[]
 for patch in build['bundles']:
  row=patch['replacement'];archive=o.ROOT/row['archive'];assert hashlib.sha256(archive.read_bytes()).hexdigest()==row['archive_sha256']
  assert hashlib.sha256((o.ROOT/row['task']).read_bytes()).hexdigest()==row['task_sha256']
  b,n=row['code_uri'][5:].split('/',1);blob=c.bucket(b).blob(n)
  if not blob.exists():blob.upload_from_filename(archive,if_generation_match=0,timeout=300)
  blob.reload();assert hashlib.sha256(blob.download_as_bytes(if_generation_match=blob.generation,timeout=300)).hexdigest()==row['archive_sha256']
  receipts.append(dict(uri=row['code_uri'],sha256=row['archive_sha256'],generation=blob.generation));print('Verified bundle',row['model'],flush=True)
 # Pause only the blocked queue manager while replacing its immutable receipts.
 subprocess.run(['systemctl','--user','stop','skyrl-circuit10-queue.service'],check=True,timeout=30)
 with (o.HERE/'circuit-queue.lock').open('a') as qlock:
  fcntl.flock(qlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert hashlib.sha256((o.HERE/'prepared.json').read_bytes()).hexdigest()==build['original_manifest_sha256']
  current=json.loads((o.HERE/'circuit-queue-state.json').read_text());assert current==state
  save(OUT/'previous-prepared.json',json.loads((o.HERE/'prepared.json').read_text()))
  save(OUT/'previous-state.json',state)
  uploads=json.loads((o.HERE/'uploads.json').read_text());existing={r['uri'] for r in uploads}
  uploads.extend(r for r in receipts if r['uri'] not in existing)
  save(o.HERE/'uploads.json',uploads)
  save(o.HERE/'prepared.json',json.loads((OUT/'prepared.json').read_text()))
  state.update(phase='retry');save(o.HERE/'circuit-queue-state.json',state)
  save(o.HERE/'circuit-checkpoint-repair.json',dict(time=time.time(),previous_attempt=5,
       action='Resume Gemma optimizer and search from step 1 with bounded checkpoint-upload retries; retain Muse then Qwen order',bundles=build['bundles'],uploads=receipts))
 subprocess.run(['systemctl','--user','start','skyrl-circuit10-queue.service'],check=True,timeout=30)
 print('Circuit queue resumed for Gemma attempt 6 from checkpoint 1',flush=True)
