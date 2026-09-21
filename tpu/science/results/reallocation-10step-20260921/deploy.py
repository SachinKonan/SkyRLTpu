"""Exact-ID, receipt-backed cutover for the approved ten-step campaign."""
import argparse,concurrent.futures,fcntl,hashlib,importlib.util,json,os,re,time
from pathlib import Path
s=importlib.util.spec_from_file_location('ops',str(Path(__file__).with_name('operations.py')));o=importlib.util.module_from_spec(s);s.loader.exec_module(o)
os.environ.update(o.ops.env)
from google.cloud import storage
M=json.loads((o.HERE/'prepared.json').read_text());ACTIVE={'RUNNING','RECOVERING','PENDING','STARTING','SUBMITTED','CANCELLING'}
def save(path,value):
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)
def verified():
 receipts={r['uri']:r for r in json.loads((o.HERE/'uploads.json').read_text())}
 for row in M['jobs']:
  assert hashlib.sha256((o.ROOT/row['archive']).read_bytes()).hexdigest()==row['archive_sha256']
  assert hashlib.sha256((o.ROOT/row['task']).read_bytes()).hexdigest()==row['task_sha256']
  assert receipts[row['code_uri']]['sha256']==row['archive_sha256']
 assert all(r['verified'] for r in json.loads((o.HERE/'preflight.json').read_text()))
def upload():
 c=storage.Client(project='vision-mix');assert c._credentials.service_account_email=='289186856710-compute@developer.gserviceaccount.com'
 receipts=[]
 for row in M['jobs']:
  path=o.ROOT/row['archive'];assert hashlib.sha256(path.read_bytes()).hexdigest()==row['archive_sha256']
  b,n=row['code_uri'][5:].split('/',1);blob=c.bucket(b).blob(n)
  if not blob.exists():blob.upload_from_filename(path,if_generation_match=0,timeout=300)
  blob.reload();assert hashlib.sha256(blob.download_as_bytes(if_generation_match=blob.generation,timeout=300)).hexdigest()==row['archive_sha256']
  receipts.append(dict(uri=row['code_uri'],sha256=row['archive_sha256'],generation=blob.generation));print('uploaded',row['label'],flush=True)
 save(o.HERE/'uploads.json',receipts)
def cancel_training():
 verified();before={r['job_id']:r for r in json.loads((o.OUT/'queue-before-cutover.json').read_text())}
 jobs=sorted({j for r in M['jobs'] if r['kind']!='farm' for j in r['supersedes']}|set(M['defer_jobs']))
 rows={r['job_id']:r for r in o.queue()};save(o.HERE/'before-cancellation.json',[rows[j] for j in jobs if j in rows])
 for j in jobs:
  assert rows[j]['run_id']==before[j]['run_id'],('job identity changed',j)
 # Cloud objects stay in their original namespaces. Preserve recovery metadata
 # before the replacement writes any newer client/database snapshot.
 c=storage.Client(project='vision-mix');snapshots=[]
 for row in M['jobs']:
  if row['kind']=='farm':continue
  bucket=c.bucket(row['bucket'][5:]);prefix='ray-training/'+row['run_id']+'/'
  objs=list(c.list_blobs(bucket,prefix=prefix+'client/tinker_log/'+row['run_id']+'/'))
  names=[b.name for b in objs if b.name.endswith(('checkpoints.jsonl','metrics.jsonl'))]
  pools=sorted(b.name for b in objs if '/puct_sampler_step_' in b.name and b.name.endswith('.json'))
  names+=pools[-1:]+[prefix+'tinker-backup.db']
  for name in names:
   blob=bucket.get_blob(name)
   if not blob:continue
   dest='migration-snapshots/reallocation-10step-20260921/'+name
   target=bucket.blob(dest)
   if not target.exists():bucket.copy_blob(blob,bucket,dest,if_generation_match=0,if_source_generation_match=blob.generation,timeout=180)
   snapshots.append(dict(source=name,generation=blob.generation,destination='gs://'+bucket.name+'/'+dest))
 save(o.HERE/'recovery-snapshots.json',snapshots)
 for j in jobs:
  if rows[j]['status'] in ACTIVE:
   text=o.sky('jobs','cancel',str(j),'--yes');(o.HERE/f'cancel-{j}.txt').write_text(text);print('cancel requested',j,flush=True)
def launch(row):
 path=o.HERE/'launches.json';receipts=json.loads(path.read_text()) if path.exists() else []
 existing=[r for r in receipts if r['label']==row['label']]
 if existing:print('already recorded',row['label'],existing[-1].get('job_id'),flush=True);return
 rows=o.queue()
 assert not any(r['run_id']==row['run_id'] and r['status'] in ACTIVE for r in rows),('duplicate active run',row['run_id'])
 receipt=dict(label=row['label'],run_id=row['run_id'],kind=row['kind'],model=row['model'],pool=row['pool'],state='submitting',time=time.time(),code_uri=row['code_uri'])
 receipts.append(receipt);save(path,receipts)
 text=o.sky('jobs','launch',str(o.ROOT/row['task']),'--pool',row['pool'],'--yes','--detach-run')
 (o.HERE/(row['label']+'-submission.txt')).write_text(text)
 ids=re.findall(r'(?:Job ID|Job id|job ID|job id)[:\s]+(\d+)',text)
 if not ids:raise RuntimeError('submission outcome uncertain; reconcile receipt before retry')
 receipt.update(job_id=int(ids[-1]),state='submitted');save(path,receipts);print('SUBMITTED',receipt['job_id'],row['kind'],row['model'],row['pool'],flush=True)
def farms():
 verified()
 inventory={r['job_id']:r for r in json.loads((o.HERE/'farm-inventory.json').read_text())}
 for row in [r for r in M['jobs'] if r['kind']=='farm']:
  old=row['supersedes'][0];current=next(r for r in o.queue() if r['job_id']==old)
  if current['status']=='RUNNING':
   assert current['cluster']==inventory[old]['cluster']
   code='''import json,urllib.request
base='http://127.0.0.1:24800'
def call(path,data=None):
 q=urllib.request.Request(base+path,data=json.dumps(data).encode() if data else None,headers={'Content-Type':'application/json'})
 with urllib.request.urlopen(q,timeout=15) as r:return json.load(r)
s=call('/status');assert s['active']==0 and not s.get('updating') and s['state'] in ('unleased','expired'), 'farm busy'
l=call('/acquire_lease',{'owner_run':'reallocation-10step-maintenance','ttl_seconds':300})
s=call('/status');assert s['active']==0 and s['lease_id']==l['lease_id'];print(json.dumps({'state':s['state'],'owner':s['owner_run']}))
'''
   print('maintenance',old,o.ssh(current['cluster'],code).strip(),flush=True)
   (o.HERE/f'cancel-{old}.txt').write_text(o.sky('jobs','cancel',str(old),'--yes'))
  elif current['status'] in ACTIVE:
   (o.HERE/f'cancel-{old}.txt').write_text(o.sky('jobs','cancel',str(old),'--yes'))
  launch(row)
def training():
 verified();rows={r['job_id']:r for r in o.queue()}
 old={j for r in M['jobs'] if r['kind']!='farm' for j in r['supersedes']}|set(M['defer_jobs'])
 assert not any(rows[j]['status'] in ACTIVE for j in old), 'old jobs have not finished cancellation'
 for kind in ['ac2','rglru','qubit']:
  for row in [r for r in M['jobs'] if r['kind']==kind]:launch(row)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['upload','cancel-training','farms','training']);a=p.parse_args()
 lock=(o.OUT/'deploy.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 dict(upload=upload,**{'cancel-training':cancel_training,'farms':farms,'training':training})[a.action]()
