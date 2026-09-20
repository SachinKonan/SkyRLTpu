"""Submit the six explicitly requested pool jobs without touching existing jobs."""
import datetime,hashlib,json,os,re,subprocess
from pathlib import Path
import google.auth
from google.auth.transport.requests import Request
from google.cloud import storage
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
SA='289186856710-compute@developer.gserviceaccount.com'
creds,_=google.auth.default(scopes=['https://www.googleapis.com/auth/cloud-platform']);creds.refresh(Request());assert creds.service_account_email==SA
active=subprocess.check_output(['/scratch/gpfs/ZHUANGL/sk7524/google-cloud-sdk/bin/gcloud','auth','list','--filter=status:ACTIVE','--format=value(account)'],text=True).strip();assert active==SA
c=storage.Client(project='vision-mix',credentials=creds)
def save(p,r):
 tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(r,f,indent=2);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def upload(uri,path,sha):
 b,n=uri[5:].split('/',1);blob=c.bucket(b).blob(n)
 if not blob.exists():blob.upload_from_filename(path,if_generation_match=0,timeout=180)
 assert hashlib.sha256(blob.download_as_bytes(timeout=180)).hexdigest()==sha
rows=json.loads((HERE/'jobs.json').read_text())['jobs'];receipts=[]
for r in rows:
 folder=ROOT/r['package_dir'];p=folder/'submission.json'
 if p.exists():
  saved=json.loads(p.read_text());assert saved['state']=='submitted','Uncertain dispatch: reconcile before retry';receipts.append(saved);continue
 archive=ROOT/r['archive'];task=ROOT/r['task_yaml'];assert hashlib.sha256(archive.read_bytes()).hexdigest()==r['archive_sha256'];assert hashlib.sha256(task.read_bytes()).hexdigest()==r['yaml_sha256']
 bucket=r['code_uri'][5:].split('/',1)[0];prefix='ray-training/'+r['run_id']+'/'
 existing=list(c.list_blobs(bucket,prefix=prefix,max_results=2));allowed=r.get('seed_destination','')[5:].split('/',1)[-1]
 assert all(b.name==allowed for b in existing),'Destination already has a workload history'
 upload(r['code_uri'],archive,r['archive_sha256'])
 if r.get('seed_file'):upload(r['seed_destination'],ROOT/r['seed_file'],r['seed_file_sha256'])
 rec=dict(run_id=r['run_id'],model=r['model'],task=r['task'],pool=r['pool'],retained=r['retained'],code_uri=r['code_uri'],archive_sha256=r['archive_sha256'],state='attempting',submitted_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
 save(p,rec)
 result=subprocess.run([str(ROOT.parent/'SkyRLTpu-multihost/third_party/TPUSwarm/.venv/bin/sky'),'jobs','launch','-p',r['pool'],str(task),'--priority',str(r['priority']),'-y','-d'],capture_output=True,text=True,timeout=360)
 log=result.stdout+result.stderr;(folder/'submit.log').write_text(log);match=re.search(r'Managed Job ID:\s*(\d+)',log)
 rec.update(exit_code=result.returncode,state='submitted' if match else 'unreconciled')
 if match:rec['job_id']=int(match.group(1))
 save(p,rec);receipts.append(rec);save(HERE/'submissions.json',{'jobs':receipts});print(json.dumps(rec),flush=True)
 assert result.returncode==0 and match,'Dispatch needs reconciliation'
