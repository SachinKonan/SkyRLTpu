"""Authorized qcmp campaign only: health-gated submission and audited recovery.

Never changes farms, pool size, other jobs, loss, bootstrap rules, or run limits.
Qwen3.5 uses historical results. Qwen3.8 bootstraps are admitted as compatible
farms become available, without waiting for the Erdos pilot. A STOP file stops this watcher.
"""
import argparse
import asyncio
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
POOL = 'tpuswarm-v6e8-east5b'
FARM_POOLS = ('tpuswarm-v6e8-asia-ne1b', 'tpuswarm-v5p32-east5a-erdos')
GROUP = 'qcmp-cross-region-20261006'
BUCKET = 'sk7524-tinker-tpu-us-east5'
ACTIVE = {'PENDING','STARTING','RUNNING','RECOVERING'}
MANIFEST = ROOT/'tpu/swarm/ray_train/campaigns/qwen35-qwen38-20261006.json'
STATE = HERE/'admission-state.json'


def save(path, obj):
    path=Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp=path.with_suffix('.tmp')
    with temp.open('w') as stream:
        json.dump(obj,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temp.replace(path)


def event(name, **fields):
    row=dict(time=time.time(),event=name,**fields)
    with (HERE/'admission-events.jsonl').open('a') as stream:
        stream.write(json.dumps(row)+'\n')
    print(json.dumps(row),flush=True)


def decision(rows, jobs, gates, farms, free_workers, max_active=8):
    if free_workers < 1 or sum(j['status'] in ACTIVE for j in jobs.values()) >= max_active:
        return None
    for row in rows:
        name=row['run_id']
        if name in jobs or row['model']!='Qwen/Qwen3.8-27B' or row.get('pool',POOL)!=POOL:continue
        # A submitted job still bringing up its ingress has not claimed farms yet.
        # Do not allocate another same-model trainer during that handoff.
        if any(j.get('pool',POOL)==POOL and j['model']==row['model'] and
               (j['status'] in {'PENDING','STARTING','RECOVERING'} or
                (j['status']=='RUNNING' and not j.get('leased')))
               for j in jobs.values()):continue
        if len(farms.get(row['model'],[]))>=1:return row
    return None


REMOTE_MONITOR = r'''
from pathlib import Path
import json, re, sqlite3, time, urllib.request
name=RUN_ID
run=Path.home()/'.cache'/name/'runs'/name
out={'run_id':name,'time':time.time(),'run_exists':run.is_dir()}
def load(path):
 try:return json.loads(path.read_text())
 except (ValueError,OSError):return None
def lines(path, maximum=4000000):
 if not path.exists():return []
 with path.open('rb') as f:
  size=path.stat().st_size;offset=max(0,size-maximum);f.seek(offset)
  data=f.read().decode(errors='replace').splitlines()
 if offset:data=data[1:]
 result=[]
 for line in data:
  try:result.append(json.loads(line))
  except ValueError:pass
 return result
bootstrap=run/'client/bootstrap'
out['bootstrap']=load(bootstrap/'complete.json')
out['bootstrap_shortfall']=load(bootstrap/'insufficient-valid-seeds.json')
out['bootstrap_generated_groups']=len(list((bootstrap/'drafts').glob('group-*/generation.json')))
out['bootstrap_graded']=len(list((bootstrap/'drafts').glob('group-*/grade-*.json')))
checkpoints=[]
for p in (run/'client/tinker_log'/name).glob('member_*/checkpoints.jsonl'):
 checkpoints += [r for r in lines(p) if r.get('state_path') and isinstance(r.get('batch'),int)]
checkpoints.sort(key=lambda r:r['batch'])
out['checkpoint']=checkpoints[-1] if checkpoints else None
out['saved_steps']=out['checkpoint']['batch'] if out['checkpoint'] else 0
database=run/'tinker.db'
if database.exists():
 try:
  connection=sqlite3.connect(database.as_uri()+'?mode=ro',uri=True,timeout=3)
  connection.row_factory=sqlite3.Row
  try:
   out['training_requests']=[dict(row) for row in connection.execute(
    'SELECT request_type,status,COUNT(*) AS count FROM futures GROUP BY request_type,status')]
   latest=connection.execute("SELECT request_id,request_type,completed_at FROM futures WHERE status='COMPLETED' AND request_type IN ('FORWARD_BACKWARD','OPTIM_STEP') ORDER BY completed_at DESC LIMIT 1").fetchone()
   out['latest_training_completion']=dict(latest) if latest else None
   oldest=connection.execute("SELECT request_id,request_type,created_at FROM futures WHERE status='PENDING' AND request_type IN ('FORWARD_BACKWARD','OPTIM_STEP') ORDER BY created_at LIMIT 1").fetchone()
   out['oldest_pending_training_request']=dict(oldest) if oldest else None
  finally:connection.close()
 except (sqlite3.Error,OSError) as exc:
  out['training_database_error']=type(exc).__name__+': '+str(exc)[:200]
trainer_log=run/'trainer.log'
out['optimizer_updates_in_tail']=0
out['latest_backward_tile']=None
if trainer_log.exists():
 with trainer_log.open('rb') as f:
  f.seek(max(0,trainer_log.stat().st_size-4000000))
  tail=f.read()
  out['optimizer_updates_in_tail']=tail.count(b'Applied optimizer step for model')
  tiles=re.findall(rb'(?m)^.*\bfb tile [0-9]+/[0-9]+[^\n]*(?:\n[ \t]*shape=[^\n]*)?',tail)
  if tiles:out['latest_backward_tile']=tiles[-1].decode(errors='replace').strip()
records=lines(run/'inference-events.jsonl')
commits=[r for r in records if r.get('event')=='adapter_committed' and re.search(r'_0*[1-9][0-9]*$',r.get('version',''))]
last=commits[-1] if commits else None
out['updated_adapter_commit']=last
out['post_update_generation']=next((r for r in reversed(records) if r.get('event') in ('borrow_generated','hybrid_generated') and last and r['time']>last['time']),None)
out['local_completions_in_tail']=sum(r.get('event')=='hybrid_generated' and str(r.get('route','')).startswith('local') for r in records)
out['borrowed_completions_in_tail']=sum(r.get('event')=='borrow_generated' for r in records)
controller=lines(run/'controller.jsonl')
out['last_controller_events']=controller[-3:]
try:
 with urllib.request.urlopen('http://127.0.0.1:24800/status',timeout=8) as f:s=json.load(f)
 out['serving']={k:s[k] for k in ('active','updating','committed','borrowing','remote_only','grading') if k in s}
except Exception as exc:out['serving_error']=type(exc).__name__
print(json.dumps(out))
'''


class Watch:
    def __init__(self, execute):
        self.execute=execute
        sys.path.insert(0,str(ROOT))
        helper=Path('/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-qubit-parallel-relaunch/tpu/science/ops/routing_relaunch_watch.py')
        spec=importlib.util.spec_from_file_location('qcmp_ops_helper',helper)
        mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
        self.ops=mod.Rollout()
        from tpu.swarm.ray_train.borrowing_supervisor import rpc, tick
        self.rpc=rpc
        self.discovery_tick=tick
        self.ssh_dir=Path(self.ops.ops.ops.env['HOME'])/'.sky/generated/ssh'
        manifest=json.loads(MANIFEST.read_text())
        assert manifest['pool']==POOL and len(manifest['runs'])==8
        assert manifest['models']==['Qwen/Qwen3.8-27B']
        self.rows=json.loads((HERE/'bundles/index.json').read_text())
        assert {r['run_id'] for r in self.rows}=={r['run_id'] for r in manifest['runs']+manifest.get('bootstrap_runs',[])}
        self.manifest=manifest
        self.admit_training=manifest.get('training_admission',True)
        order={'erdos':0,'ac1':1,'circuit':2,'qubit':3}
        self.rows.sort(key=lambda r:(r['seed'],order[r['problem']],r['model']))
        for row in self.rows:
            from tpu.swarm.ray_train.config import Config
            c=Config.load(ROOT/row['profile'])
            row.update(bucket=c.bucket.removeprefix('gs://'),zone=c.effective_zone,accelerator=c.accelerator)
        self.by_name={r['run_id']:r for r in self.rows}
        self.state=json.loads(STATE.read_text()) if STATE.exists() else dict(jobs={},intents={},pilot_gates={})
        self.identity=hashlib.sha256(MANIFEST.read_bytes()+(HERE/'bundles/index.json').read_bytes()).hexdigest()
        if self.state.get('identity') not in (None,self.identity):raise RuntimeError('Campaign artifacts changed; reconcile admission state')
        self.state.update(identity=self.identity,pid=os.getpid(),host=socket.gethostname(),execute=execute,worktree=str(ROOT),pool=POOL)

    def persist(self):
        self.state['updated']=time.time();save(STATE,self.state)

    def inventory(self):
        rows=self.ops.query("i.name LIKE 'qcmp-%' OR ((i.pool IN (?,?) OR i.name LIKE 'inference-farm-%qwen38%') AND s.status IN ('RUNNING','STARTING','PENDING','RECOVERING'))",FARM_POOLS)
        jobs={}
        for row in rows:
            if row['name'] in self.by_name:
                if row['name'] in jobs:raise RuntimeError('Duplicate campaign job name')
                if row['pool']!=self.by_name[row['name']].get('pool',POOL):raise RuntimeError('Campaign job in unexpected pool')
                jobs[row['name']]=dict(row,model=self.by_name[row['name']]['model'])
        return rows,jobs

    def farm_inventory(self, rows):
        # Restrict updates to our manifest identities, while accepting farms
        # across both authorized pools. Existing leases remain with their owners.
        scoped=[dict(row,run_id=row['name']) for row in rows
                if row['name'] in self.by_name or
                ((row['pool'] in FARM_POOLS or 'qwen38' in row['name']) and 'inference-farm' in row['name']
                 and ('qwen35' in row['name'] or 'qwen38' in row['name']))]
        trainer_ids=[r['job_id'] for r in scoped if r['name'] in self.by_name]
        def call(cluster,action,body=None):
            return self.rpc(self.ssh_dir,cluster,action,24800,body,group=GROUP)
        discovery=self.discovery_tick(scoped,None,trainer_ids,call,
                                      dry_run=not self.execute,run_scoped_only=True)
        self.state['discovery']=discovery
        available={model:[] for model in ('Qwen/Qwen3.5-27B','Qwen/Qwen3.8-27B')}
        for farm in discovery['farms']:
            if farm.get('state') not in ('unleased','expired') or farm.get('active',0):continue
            for model in farm['models']:
                if model in available:available[model].append(farm)
        self.state['farm_inventory']={model:dict(free=len(farms),farms=farms)
                                      for model,farms in available.items()}
        return available

    def monitor(self, job):
        if job['status']!='RUNNING' or not job.get('cluster'):return None
        try:
            code='RUN_ID = '+repr(job['name'])+'\n'+REMOTE_MONITOR
            result=json.loads(self.ops.ops.ssh(job['cluster'],code,timeout=35))
            cp=result.get('checkpoint')
            if cp:
                match=re.fullmatch(r'tinker://([^/]+)/weights/([^/]+)',cp['state_path'])
                if match:
                    key=f"ray-training/{job['name']}/checkpoints/{match[1]}/{match[2]}.tar.gz"
                    bucket=self.by_name[job['name']]['bucket']
                    blob=self.ops.storage.bucket(bucket).get_blob(key,timeout=15)
                    result['durable_checkpoint']=dict(uri=f'gs://{bucket}/{key}',exists=bool(blob),size=blob.size if blob else None)
            save(HERE/'monitor'/f"{job['name']}.json",result)
            return result
        except Exception as exc:return dict(monitor_error=type(exc).__name__+': '+str(exc)[-500:])

    def ready_workers(self):
        # Read the same local controller state without a potentially stalled
        # API request. Provider health and managed-job claims remain required.
        from sky.serve import serve_state
        replicas=serve_state.get_replica_infos(POOL)
        claimed={j['cluster'] for j in self.ops.query(
            "i.pool=? AND s.status IN ('PENDING','STARTING','RUNNING','RECOVERING')", (POOL,))
            if j.get('cluster')}
        ids={str(r.replica_id) for r in replicas
             if r.status.value=='READY' and r.cluster_name not in claimed}
        nodes=json.loads(self.ops.ops.run(['gcloud','compute','tpus','tpu-vm','list','--project','vision-mix','--zone','us-east5-b','--format=json'],timeout=60))
        healthy=[]
        for node in nodes:
            match=re.search(re.escape(POOL)+r'-(\d+)-',node['name'])
            if match and match[1] in ids and node.get('acceleratorType')=='v6e-8' and node.get('state')=='READY' and node.get('health')=='HEALTHY':
                healthy.append(dict(worker=match[1],node=node['name'],cluster=node.get('labels',{}).get('skypilot-cluster-name')))
        self.state['free_workers']=healthy
        return healthy

    def submit(self,row,farms):
        import yaml
        name=row['run_id'];task=Path(row['task']);archive=Path(row['archive']);pool=row.get('pool',POOL)
        existing=self.ops.query('i.name=?',(name,))
        if existing:
            if len(existing)!=1 or existing[0]['pool']!=pool:raise RuntimeError('Ambiguous existing job '+name)
            self.state['jobs'][name]=existing[0];self.persist();return
        if name in self.state['intents']:raise RuntimeError('Unresolved launch intent: '+name)
        if row.get('requires_shared_pool') and not row.get('shared_pool_ready'):
            raise RuntimeError('Shared bootstrap is not materialized')
        doc=yaml.safe_load(task.read_text())
        assert doc['name']==name and doc['resources']['zone']==row['zone']
        assert doc['resources']['accelerators']==row['accelerator']
        assert row['uri'].startswith('gs://'+row['bucket']+'/')
        assert doc['envs']['RAY_TRAIN_CODE']==row['uri']
        assert hashlib.sha256(archive.read_bytes()).hexdigest()==row['sha256']==doc['envs']['RAY_TRAIN_CODE_SHA256']
        # Verify the exact profile inside the submitted bundle, not just the worktree.
        import tarfile
        from tpu.swarm.ray_train.config import Config
        from tpu.swarm.ray_train.regional_preflight import check_qwen38
        with tarfile.open(archive) as bundle:
            matches=[m for m in bundle.getmembers() if m.name.endswith('/profiles/'+Path(row['profile']).name)]
            if len(matches)!=1:raise RuntimeError('Missing or ambiguous bundled launch profile')
            bundled=Config.from_dict(json.load(bundle.extractfile(matches[0])))
        if bundled.run_id!=name:raise RuntimeError('Bundled run identity changed')
        proof=check_qwen38(bundled,self.ops.storage,task=doc)
        save(HERE/'monitor'/(name+'-regional-preflight.json'),dict(time=time.time(),archive_sha256=row['sha256'],**proof))
        self.ops.upload(archive,row['uri'])
        self.state['intents'][name]=dict(time=time.time(),task=str(task),task_sha256=hashlib.sha256(task.read_bytes()).hexdigest(),farms=farms)
        self.persist();event('submitting',run_id=name,pool=pool)
        output=self.ops.ops.sky('jobs','launch',str(task),'--pool',pool,'--yes','--detach-run')
        (HERE/(name+'-launch.txt')).write_text(output)
        ids=re.findall(r'(?:Job ID|Job id|job ID|job id)[:\s]+(\d+)',output)
        if not ids:raise RuntimeError('Ambiguous submission output; inspect intent and queue')
        self.state['jobs'][name]=dict(job_id=int(ids[-1]),status='PENDING',name=name,pool=pool,model=row['model'])
        self.persist();event('submitted',run_id=name,job_id=int(ids[-1]),pool=pool)

    def recover_claims(self, jobs):
        import sqlite3
        import filelock
        from sky.serve import serve_utils
        from sky.jobs import state as job_state
        from tpu.swarm.ray_train.pool_recovery import release_stale_claim, REMOTE_ATTEMPT_AUDIT

        def release_for_retry(job_id):
            # A resumed RECOVERING controller assumes it has a cluster and
            # asserts before recovery when the claim is cleared. Mark the
            # audited dead attempt pending first so its next loop launches.
            asyncio.run(job_state.set_backoff_pending_async(
                job_id, 0, reason='Audited terminal worker attempt; requeue before releasing stale pool claim.'))
            job_state.set_current_cluster_name(job_id, None)

        def read_job(job_id):
            with sqlite3.connect('file:'+str(self.ops.db)+'?mode=ro',uri=True) as db:
                db.row_factory=sqlite3.Row
                row=db.execute('SELECT i.name,i.pool,i.current_cluster_name AS cluster,'
                               'i.job_id_on_pool_cluster AS worker_job_id,s.status,'
                               's.recovering_from_failure FROM job_info i JOIN spot s '
                               'ON i.spot_job_id=s.spot_job_id WHERE i.spot_job_id=?',
                               (job_id,)).fetchone()
                return dict(row) if row else None

        for job in jobs.values():
            if job['status'] not in {'PENDING','RECOVERING'} or not job.get('cluster'):
                continue
            # A removed SSH config means the provider worker has disappeared;
            # leave its ordinary provisioning recovery to SkyPilot.
            if not (self.ssh_dir/job['cluster']).is_file():
                continue
            try:
                proof=release_stale_claim(
                    job['job_id'],set(self.by_name),read_job=read_job,
                    claimants=job_state.get_nonterminal_job_ids_by_pool,
                    inspect_worker=lambda c: json.loads(self.ops.ops.ssh(c,REMOTE_ATTEMPT_AUDIT,timeout=25)),
                    release=release_for_retry,
                    pool_lock=lambda p: filelock.FileLock(serve_utils.get_service_filelock_path(p),timeout=2),
                    now=time.time)
                if proof:
                    save(HERE/'monitor'/(str(job['job_id'])+'-stale-claim-released.json'),dict(time=time.time(),**proof))
                    event('stale_claim_released',**proof)
            except Exception as exc:
                event('recovery_audit_deferred',job_id=job['job_id'],detail=str(exc)[-300:])

    def tick(self):
        rows,jobs=self.inventory()
        if self.execute:
            self.recover_claims(jobs)
            rows,jobs=self.inventory()
        with ThreadPoolExecutor(max_workers=5) as pool:
            observations=list(pool.map(self.monitor,jobs.values()))
        for job,observed in zip(jobs.values(),observations):
            if observed:
                borrowing=(observed.get('serving') or {}).get('borrowing') or {}
                job.update(saved_steps=observed.get('saved_steps',0),leased=bool(borrowing.get('held') or borrowing.get('reserved')),monitor=observed)
                if len(borrowing.get('held',[]))>2:event('lease_limit_exceeded',run_id=job['name'],held=borrowing['held'])
                boot=observed.get('bootstrap') or {}
                gate=(1<=boot.get('retained',0)<=512 and boot.get('drafted')==1024 and observed.get('saved_steps',0)>=1
                      and observed.get('optimizer_updates_in_tail',0)>0 and observed.get('post_update_generation')
                      and (observed.get('durable_checkpoint') or {}).get('exists'))
                if gate and not self.state['pilot_gates'].get(job['name']):
                    self.state['pilot_gates'][job['name']]=True
                    save(HERE/'monitor'/(job['name']+'-first-step-proof.json'),observed)
                    event('first_step_verified',run_id=job['name'],job_id=job['job_id'])
        self.state['jobs']=jobs
        farms=self.farm_inventory(rows)
        if self.execute and self.admit_training:
            from shared_pool_admission import materialize, promote_capped_journal
            sources={r['run_id']:r for r in self.manifest['runs']+self.manifest.get('bootstrap_runs',[])}
            for source_id in self.manifest['bootstrap_sources'].values():
                result=promote_capped_journal(self,sources[source_id],ROOT,HERE)
                if result:event('capped_bootstrap_promoted',run_id=source_id,retained=result['retained'],drafted=result['drafted'])
            changed=False
            for row in self.rows:
                if row.get('requires_shared_pool') and row['run_id'] not in jobs:
                    changed=materialize(self,row,self.manifest,ROOT,HERE) or changed
            if changed:
                save(HERE/'bundles/index.json',self.rows)
                self.identity=hashlib.sha256(MANIFEST.read_bytes()+(HERE/'bundles/index.json').read_bytes()).hexdigest()
                self.state['identity']=self.identity
                self.persist()
        # Avoid repeated provider calls while there is no matching farm at all.
        ready=self.ready_workers() if self.admit_training and any(farms.values()) else []
        chosen=decision(self.rows,jobs,self.state['pilot_gates'],farms,len(ready)) if self.admit_training else None
        # Hybrid consumers can queue without a remote farm after their shared pool
        # is immutable and local. Six v4 consumers plus two East5 trainers total.
        if self.admit_training and sum(j['status'] in ACTIVE for j in jobs.values())<8:
            chosen=next((r for r in self.rows if r.get('shared_pool_ready') and r['run_id'] not in jobs),chosen)
        self.state.update(phase='admission_ready' if chosen else 'waiting_for_farm_or_worker',
                          next_run=chosen['run_id'] if chosen else None)
        self.persist()
        event('tick',phase=self.state['phase'],farms={m:len(v) for m,v in farms.items()},
              jobs={n:dict(job_id=j['job_id'],status=j['status'],saved_steps=j.get('saved_steps'),leased=j.get('leased')) for n,j in jobs.items()},
              next_run=self.state['next_run'],execute=self.execute)
        if chosen and self.execute:self.submit(chosen,farms[chosen['model']])
        return len(jobs)==len(self.rows) and all(j['status'] not in ACTIVE for j in jobs.values())


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--execute',action='store_true');parser.add_argument('--once',action='store_true')
    args=parser.parse_args()
    with (HERE/'admission.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        watcher=Watch(args.execute)
        event('watch_started',pid=os.getpid(),execute=args.execute)
        while not (HERE/'STOP').exists():
            try:
                if watcher.tick():event('campaign_terminal');break
            except Exception as exc:
                event('watch_error',error=type(exc).__name__,detail=str(exc)[-1500:])
            if args.once:break
            time.sleep(45)

if __name__=='__main__':main()
