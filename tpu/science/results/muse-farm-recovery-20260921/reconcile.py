import importlib.util,json,time,sqlite3
from pathlib import Path
s=importlib.util.spec_from_file_location('o','.science/reallocation-10step/ops.py');o=importlib.util.module_from_spec(s);s.loader.exec_module(o)
d=Path('.science/muse-farm-memory-fix-20260921')
assert not (d/'reservation-reconcile.json').exists(),'reconcile existing action'
q=o.queue();m=next(r for r in q if r['job_id']==1368);g=next(r for r in q if r['job_id']==1369)
assert m['status']=='PENDING' and m['cluster']=='tpuswarm-v4-32-central2-smoke-177'
assert g['status']=='PENDING' and g['cluster'] is None
(d/'reservation-reconcile.json').write_text(json.dumps({'time':time.time(),'state':'checking','muse':m,'gemma':g},indent=2))
# Reconcile the uncertain POST against persistent API records and the worker.
query="""from sky.skylet import job_lib; import json; rs=job_lib.load_job_queue(job_lib.dump_job_queue(None,True)); print(json.dumps([{k:r.get(k) for k in ['job_id','job_name','status','submitted_at']} for r in rs],default=str))"""
remote='import subprocess\np=subprocess.run(["/home/gcpuser/skypilot-runtime/bin/python","-c",'+repr(query)+'],capture_output=True,text=True,timeout=30)\nassert p.returncode==0,p.stderr\nprint(p.stdout)'
rows=json.loads(o.ssh(m['cluster'],remote));assert all(r['status'] in ['JobStatus.FAILED','JobStatus.CANCELLED','JobStatus.SUCCEEDED'] for r in rows),rows
assert max(r['submitted_at'] for r in rows)<1789970000
(d/'worker-no-new-job.json').write_text(json.dumps(rows,indent=2))
p=Path(o.ops.env['HOME'])/'.sky/api_server/requests.db'
c=sqlite3.connect('file:'+str(p)+'?mode=ro',uri=True,timeout=10)
rs=c.execute('select request_id,status,created_at,finished_at from requests where created_at>? and cluster_name=? and name=?',(1789974000,m['cluster'],'sky.exec')).fetchall()
assert all(r[1]=='CANCELLED' for r in rs),rs
(d/'api-no-live-exec.json').write_text(json.dumps(rs,indent=2))
# Remove lower-priority contention while the existing Muse controller retries.
(d/'cancel-gemma-1369.txt').write_text(o.sky('jobs','cancel','1369','--yes'))
for _ in range(40):
 q=o.queue();g=next(r for r in q if r['job_id']==1369)
 if g['status']=='CANCELLED':break
 time.sleep(3)
else:raise RuntimeError('Gemma cancellation not confirmed')
# Use SkyPilot's state helper under the same pool lock as its scheduler.
code="""import filelock,json
from sky.jobs import state
from sky.serve import serve_utils
pool='tpuswarm-v4-32-central2-smoke'
with filelock.FileLock(serve_utils.get_service_filelock_path(pool)):
 assert str(state.get_status(1368))=='ManagedJobStatus.PENDING'
 assert state.get_pool_submit_info(1368)==('tpuswarm-v4-32-central2-smoke-177',None)
 state.set_current_cluster_name(1368,None)
 print(json.dumps({'job_id':1368,'cleared_unlaunched_reservation':True,'submit_info':state.get_pool_submit_info(1368)}))
"""
out=o.run([o.ops.venv/'python','-c',code]);print(out);(d/'reservation-cleared.json').write_text(out)
(d/'reservation-reconcile.json').write_text(json.dumps({'time':time.time(),'state':'reservation_cleared','muse_job_id':1368,'cancelled_gemma_job_id':1369,'gemma_needs_requeue':True},indent=2))
