"""Verify durable bootstrap contracts and completed pools before same-run resume."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import tarfile
from google.cloud import storage
from tpu.swarm.ray_train.config import Config
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
client=storage.Client(project='vision-mix')
rows=json.loads((HERE/'jobs.json').read_text())['jobs']
def check(row):
 cfg=Config.load(ROOT/row['profile']);bucket,prefix=cfg.run_gcs[5:].split('/',1)
 def get(name):
  blob=client.bucket(bucket).get_blob(prefix+'/'+name)
  return (json.loads(blob.download_as_bytes()),blob.generation) if blob else (None,None)
 contract,generation=get('client/bootstrap/contract.json')
 complete,_=get('client/bootstrap/complete.json')
 record=dict(old_job_id=row['old_job_id'],run_id=row['run_id'],run_gcs=cfg.run_gcs,bootstrap_contract=bool(contract),complete=complete)
 if contract:
  assert contract['contract']['config']==cfg.to_dict(),f"Bootstrap config differs for {row['old_job_id']}"
  with tarfile.open(ROOT/row['archive']) as archive:
   data=archive.extractfile('tpu/swarm/ray_train/seed_bootstrap.py').read()
  assert hashlib.sha256(data).hexdigest()==contract['contract']['implementation_sha256']
  record['contract_generation']=generation
 if complete:
  assert contract
  pool,_=get('client/tinker_log/'+row['run_id']+'/puct_sampler_step_000000.json')
  assert pool is not None
  assert hashlib.sha256(json.dumps(pool,sort_keys=True).encode()).hexdigest()==complete['pool_sha256']
  record['retained']=len(pool['states'])
 for blob in client.list_blobs(bucket,prefix=prefix+'/client/tinker_log/'+row['run_id']+'/'):
  if blob.name.endswith('/checkpoints.jsonl'):
   lines=blob.download_as_text().splitlines()
   record.setdefault('checkpoint_journals',[]).append(dict(path=blob.name,rows=len(lines),last=json.loads(lines[-1]) if lines else None))
 print(json.dumps({k:v for k,v in record.items() if k in ('old_job_id','bootstrap_contract','retained')}),flush=True)
 return record
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(check,rows))
(HERE/'preserved.json').write_text(json.dumps({'runs':records},indent=2)+'\n')
