"""Discovery for registered on-demand trainers plus the existing Qubit job.
Never admits/cancels jobs, acquires leases, or changes farm services.
"""
import fcntl,json,os,time
from concurrent.futures import ThreadPoolExecutor
from launch_watch import Watch,HERE,save
from od_deploy import OUT,ssh
from tpu.swarm.ray_train.borrowing_supervisor import REMOTE
from tpu.swarm.ray_train.direct_discovery import tick as direct_tick, stable_completion
import od_status
NAMES={'inference-farm-rq4-qwen38-east5a-v5p32-004','inference-farm-rq4-qwen38-east5a-v5p32-005'}
def direct_call(row,body=None):
 params=dict(action='target',port=24800,query='?group='+row['group'])
 if body is not None:params['body']=body
 return json.loads(ssh(row['external_ip'],'PARAMS = '+repr(params)+'\n'+REMOTE))
def tick(w):
 # Query failure raises: never turn an inventory outage into an empty list.
 registry=json.loads((OUT/'discovery-targets.json').read_text())
 registrations=registry['trainers']
 default_pools=['tpuswarm-v5p32-east5a-erdos']
 allowed_pools={pool for row in registrations for pool in row.get('farm_pools',default_pools)}
 farms=w.ops.query("i.name LIKE '%inference-farm%qwen38%' AND s.status='RUNNING'",())
 farms=[j for j in farms if j['pool'] in allowed_pools]
 def probe(j):
  return j,w.rpc(w.ssh_dir,j['cluster'],'farm',24800)
 with ThreadPoolExecutor(max_workers=8) as pool:probes=list(pool.map(probe,farms))
 healthy=[dict(source_job=j['job_id'],source_cluster=j['cluster'],source_pool=j['pool'],**p['result']) for j,p in probes if p.get('ok')]
 # Open extra cloud capacity automatically only after the dedicated adapter
 # path has produced a completed request. Persist the exact observed proof.
 for row in registrations:
  if row.get('allow_spot') or not row.get('enable_spot_after_stable_completion'):continue
  proof_path=OUT/(row['run_id']+'-stable-completion.json')
  try:
   if proof_path.exists():
    proof=json.loads(proof_path.read_text())
    assert proof['run_id']==row['run_id'] and proof['stable_urls']==row['stable_urls']
   else:
    code="from pathlib import Path\nimport json\np=Path.home()/'.cache'/"+repr(row['run_id'])+"/'runs'/"+repr(row['run_id'])+"/'inference-events.jsonl'\nrows=[]\nif p.exists():\n with p.open('rb') as f:\n  offset=max(0,p.stat().st_size-4000000);f.seek(offset);lines=f.read().decode(errors='replace').splitlines()\n for line in (lines[1:] if offset else lines):\n  try: rows.append(json.loads(line))\n  except ValueError: pass\nprint(json.dumps(rows))\n"
    evidence=stable_completion(json.loads(ssh(row['external_ip'],code)),row['stable_urls'])
    if evidence is None:continue
    proof=dict(time=time.time(),run_id=row['run_id'],stable_urls=row['stable_urls'],evidence=evidence)
    save(proof_path,proof)
   row['allow_spot']=True
  except Exception as e:
   print(json.dumps(dict(event='stable_gate_probe_failed',run_id=row['run_id'],error=str(e))),flush=True)
 direct=[]
 for row in registrations:
  eligible=[farm for farm in healthy if farm['source_pool'] in row.get('farm_pools',default_pools)]
  direct.extend(direct_tick([row],eligible,direct_call))
 # Keep the already-running Qubit trainer's existing supervisor group intact.
 job=w.ops.job(2370);d={}
 if job['status']=='RUNNING':
  rows=[dict(job,run_id='qcmp-qwen38-qubit-pwc05-s1-shared-v464-wd-20261007')]+[dict(j,run_id=j['name']) for j in farms if j['name'] in NAMES]
  cached={j['cluster']:p for j,p in probes}
  def call(cluster,action,body=None):
   if action=='farm':return cached[cluster]
   return w.rpc(w.ssh_dir,cluster,action,24800,body,group='qcmp-v5p-sequential-20261010')
  d=w.discovery_tick(rows,None,[2370],call,run_scoped_only=True)
 report=dict(time=time.time(),job=job,farms=farms,discovery=d,direct=direct,
             unavailable=[dict(job_id=j['job_id'],detail=p) for j,p in probes if not p.get('ok')])
 save(OUT/'spot-farms.json',report)
 print(json.dumps(dict(time=report['time'],direct=direct,healthy_farms=len(healthy),unavailable=report['unavailable'])),flush=True)
 od_status.main()
def main():
 with (OUT/'watch.lock').open('a') as f:
  fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);(OUT/'watch.pid').write_text(str(os.getpid()));w=Watch(False)
  while not (OUT/'STOP').exists():
   try:tick(w)
   except Exception as e:print(json.dumps(dict(time=time.time(),error=type(e).__name__+': '+str(e))),flush=True)
   time.sleep(30)
if __name__=='__main__':main()
