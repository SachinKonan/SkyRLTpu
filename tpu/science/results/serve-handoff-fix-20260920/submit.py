"""Resume preserved run namespaces only after their old managed jobs terminate."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import google.auth
from google.auth.transport.requests import Request

ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
SA='289186856710-compute@developer.gserviceaccount.com'
creds,_=google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform']);creds.refresh(Request())
active=subprocess.check_output(['/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gcloud','auth','list','--filter=status:ACTIVE','--format=value(account)'],text=True).strip()
assert active==SA and creds.service_account_email==SA
assert (HERE/'uploaded.json').exists() and (HERE/'preserved.json').exists()
rows=json.loads((HERE/'jobs.json').read_text())['jobs']
db=sqlite3.connect('file:/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/spot_jobs.db?mode=ro',uri=True)
terminal={'CANCELLED','FAILED','FAILED_SETUP','FAILED_PRECHECKS','FAILED_NO_RESOURCE','FAILED_CONTROLLER','SUCCEEDED'}
for row in rows:
 state=db.execute('select status from spot where spot_job_id=?',(row['old_job_id'],)).fetchone()[0]
 if state not in terminal:raise RuntimeError(f"Old job {row['old_job_id']} still {state}; refuse concurrent writers")

def save(path,record):
 tmp=path.with_suffix('.tmp')
 with tmp.open('w') as stream:
  json.dump(record,stream,indent=2);stream.flush();os.fsync(stream.fileno())
 tmp.replace(path)

for row in rows:
 folder=ROOT/row['package_dir'];receipt=folder/'submission.json'
 if receipt.exists():
  previous=json.loads(receipt.read_text())
  if previous.get('state')=='submitted':print(json.dumps(previous),flush=True);continue
  raise RuntimeError('Unreconciled dispatch; check controller before retrying: '+str(receipt))
 assert hashlib.sha256((ROOT/row['archive']).read_bytes()).hexdigest()==row['archive_sha256']
 assert hashlib.sha256((ROOT/row['task_yaml']).read_bytes()).hexdigest()==row['yaml_sha256']
 record=dict(run_id=row['run_id'],old_job_id=row['old_job_id'],pool=row['pool'],priority=row['priority'],state='attempting',code_uri=row['code_uri'],archive_sha256=row['archive_sha256'],submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 save(receipt,record)
 cmd=[str(ROOT.parent/'SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/sky'),'jobs','launch','-p',row['pool'],str(ROOT/row['task_yaml']),'--priority',str(row['priority']),'-y','-d']
 p=subprocess.run(cmd,text=True,capture_output=True,timeout=360)
 log=p.stdout+p.stderr;(folder/'submit.log').write_text(log)
 match=re.search(r'Managed Job ID:\s*(\d+)',log)
 record.update(exit_code=p.returncode,state='submitted' if match else 'unreconciled')
 if match:record['job_id']=int(match.group(1))
 save(receipt,record);print(json.dumps(record),flush=True)
 if p.returncode or not match:raise RuntimeError('Dispatch needs reconciliation')
