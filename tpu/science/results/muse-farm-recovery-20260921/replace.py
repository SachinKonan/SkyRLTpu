import importlib.util,json,hashlib,re,time,fcntl
from pathlib import Path
s=importlib.util.spec_from_file_location('o','.science/reallocation-10step/ops.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o)
d=Path('.science/muse-farm-memory-fix-20260921');lock=(d/'lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def save(n,x):
 p=d/n;t=p.with_suffix('.tmp');t.write_text(json.dumps(x,indent=2)+'\n');t.replace(p)
def cancelled(id):
 for _ in range(50):
  q=o.queue();r=next(x for x in q if x['job_id']==id)
  if r['status']=='CANCELLED':return q
  time.sleep(3)
 raise RuntimeError(f'cancel {id} unconfirmed; reconcile')
def launch(row,name):
 state=dict(time=time.time(),run_id=row['run_id'],state='submitting');save(name+'.json',state)
 out=o.sky('jobs','launch',str(o.ROOT/row['task']),'--pool',row['pool'],'--yes','--detach-run');(d/(name+'.txt')).write_text(out)
 ids=re.findall(r'(?:Job ID|Job id|job ID|job id)[:\s]+(\d+)',out);assert ids,'uncertain launch; reconcile'
 state.update(state='submitted',job_id=int(ids[-1]));save(name+'.json',state);return state['job_id']
assert not (d/'intent.json').exists(),'reconcile existing intent'
r=json.loads((d/'prepared.json').read_text());assert json.loads((d/'upload.json').read_text())['md5_verified']
g=next(x for x in json.loads((o.HERE/'prepared.json').read_text())['jobs'] if x['run_id']=='farm10-gemma-2-20260921')
for row in [r,g]:assert hashlib.sha256((o.ROOT/row['task']).read_bytes()).hexdigest()==row['task_sha256']
q=o.queue();a=next(x for x in q if x['job_id']==1366);b=next(x for x in q if x['job_id']==1367)
assert a['run_id']==r['run_id'] and a['status']=='STARTING' and a['cluster']=='tpuswarm-v4-32-central2-smoke-177',a
assert b['run_id']==g['run_id'] and b['status']=='PENDING' and b['cluster'] is None,b
save('queue-before.json',q);save('intent.json',dict(time=time.time(),state='cancelling',old_muse=1366,old_gemma=1367))
# Withdraw the idle lower-priority request before freeing the occupied slot.
(d/'cancel-gemma.txt').write_text(o.sky('jobs','cancel','1367','--yes'));cancelled(1367)
(d/'cancel-muse.txt').write_text(o.sky('jobs','cancel','1366','--yes'));q=cancelled(1366)
assert not any(x['run_id'] in [r['run_id'],g['run_id']] and x['status'] in {'RUNNING','STARTING','PENDING','RECOVERING','SUBMITTED','CANCELLING'} for x in q)
save('queue-cancelled.json',q)
id=launch(r,'muse-submission')
for _ in range(60):
 q=o.queue();a=next(x for x in q if x['job_id']==id)
 if a['cluster'] and a['status'] in {'STARTING','RUNNING'}:break
 time.sleep(3)
else:raise RuntimeError('Muse not placed; requeue Gemma after reconciling')
save('muse-placed.json',a)
gid=launch(g,'gemma-submission')
q=o.queue();rows=[x for x in q if x['job_id'] in [id,gid,1366,1367]];save('queue-after.json',rows)
assert next(x for x in rows if x['job_id']==id)['run_id']==r['run_id']
assert next(x for x in rows if x['job_id']==gid)['run_id']==g['run_id']
save('intent.json',dict(time=time.time(),state='complete',muse_job_id=id,gemma_job_id=gid));print(json.dumps(rows),flush=True)
