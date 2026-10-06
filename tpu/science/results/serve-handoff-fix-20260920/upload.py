"""Verify identity and publish replacement bundles before stopping any workload."""
import hashlib
import json
from pathlib import Path
import subprocess
import google.auth
from google.auth.transport.requests import Request
from google.cloud import storage
ROOT=Path(__file__).resolve().parents[4]
HERE=Path(__file__).resolve().parent
SA='289186856710-compute@developer.gserviceaccount.com'
creds,_=google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform']);creds.refresh(Request())
active=subprocess.check_output(['/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gcloud','auth','list','--filter=status:ACTIVE','--format=value(account)'],text=True).strip()
assert active==SA and creds.service_account_email==SA
client=storage.Client(project='vision-mix',credentials=creds)
rows=json.loads((HERE/'jobs.json').read_text())['jobs']
for row in rows:
 archive=ROOT/row['archive']; task=ROOT/row['task_yaml']
 assert hashlib.sha256(archive.read_bytes()).hexdigest()==row['archive_sha256']
 assert hashlib.sha256(task.read_bytes()).hexdigest()==row['yaml_sha256']
 bucket,name=row['code_uri'][5:].split('/',1);blob=client.bucket(bucket).blob(name)
 if not blob.exists():blob.upload_from_filename(archive,if_generation_match=0,timeout=180)
 assert hashlib.sha256(blob.download_as_bytes(timeout=180)).hexdigest()==row['archive_sha256']
 print(json.dumps(dict(old_job=row['old_job_id'],uploaded=True)),flush=True)
(HERE/'uploaded.json').write_text(json.dumps({'artifacts':[{'old_job_id':r['old_job_id'],'sha256':r['archive_sha256']} for r in rows]},indent=2)+'\n')
