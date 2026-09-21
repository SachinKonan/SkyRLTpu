"""Submit only the three receipt-backed v6e qubit jobs; never cancel work."""
import fcntl,hashlib,importlib.util,json,os,re,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
s=importlib.util.spec_from_file_location('o',ROOT/'tpu/science/results/reallocation-10step-20260921/operations.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o);os.environ.update(o.ops.env)
from google.cloud import storage
ACTIVE={'RUNNING','RECOVERING','PENDING','STARTING','SUBMITTED','CANCELLING'}
def save(path,data):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(data,indent=2)+'\n');temp.replace(path)
def main():
 lock=(HERE/'submit.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 c=storage.Client(project='vision-mix');account='289186856710-compute@developer.gserviceaccount.com'
 assert c._credentials.service_account_email==account
 assert o.run(['gcloud','auth','list','--filter=status:ACTIVE','--format=value(account)']).strip()==account
 jobs=json.loads((HERE/'prepared.json').read_text())['jobs'];q=o.queue();save(HERE/'queue-before.json',q)
 for zone in ['us-central1-b','us-east5-b']:
  nodes=json.loads(o.run(['gcloud','compute','tpus','tpu-vm','list','--project','vision-mix','--zone',zone,'--format=json']));save(HERE/(zone+'-provider.json'),nodes)
 receipts_path=HERE/'submissions.json';receipts=json.loads(receipts_path.read_text()) if receipts_path.exists() else []
 for r in jobs:
  if any(x['run_id']==r['run_id'] for x in receipts):print('Already recorded; no resubmission',r['run_id'],flush=True);continue
  assert not any(x['run_id']==r['run_id'] and x['status'] in ACTIVE for x in q),'unreceipted active run'
  for field,hashfield in [('archive','archive_sha256'),('task','task_sha256'),('seed_file','seed_file_sha256')]:assert hashlib.sha256((ROOT/r[field]).read_bytes()).hexdigest()==r[hashfield]
  uploads=[]
  for uri,path,digest in [(r['code_uri'],ROOT/r['archive'],r['archive_sha256']),(r['seed_destination'],ROOT/r['seed_file'],r['seed_file_sha256'])]:
   b,n=uri[5:].split('/',1);blob=c.bucket(b).blob(n)
   if not blob.exists():blob.upload_from_filename(path,if_generation_match=0,timeout=180)
   blob.reload();assert hashlib.sha256(blob.download_as_bytes(if_generation_match=blob.generation,timeout=180)).hexdigest()==digest
   uploads.append(dict(uri=uri,generation=blob.generation,sha256=digest))
  receipt=dict(run_id=r['run_id'],pool=r['pool'],priority=r['priority'],epochs=10,state='submitting',time=time.time(),uploads=uploads)
  receipts.append(receipt);save(receipts_path,receipts)
  result=o.sky('jobs','launch',str(ROOT/r['task']),'--pool',r['pool'],'--yes','--detach-run');(HERE/(r['run_id']+'-submission.txt')).write_text(result)
  ids=re.findall(r'(?:Job ID|Job id|job ID|job id)[:\s]+(\d+)',result)
  if not ids:raise RuntimeError('Uncertain submission; reconcile before retrying')
  receipt.update(job_id=int(ids[-1]),state='submitted');save(receipts_path,receipts);print('Submitted',receipt['job_id'],r['run_id'],r['pool'],flush=True)
if __name__=='__main__':main()
