"""Ask affected controllers to save and stop before cancelling their old attempts."""
import concurrent.futures
import datetime
import json
from pathlib import Path
import sqlite3
import subprocess

ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'jobs.json').read_text())['jobs']
assert (HERE/'uploaded.json').is_file(), 'Publish every replacement first'
db=sqlite3.connect('file:/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/spot_jobs.db?mode=ro',uri=True)
for row in rows:
 state=db.execute('select s.status,j.current_cluster_name from job_info j join spot s on j.spot_job_id=s.spot_job_id where j.spot_job_id=?',(row['old_job_id'],)).fetchone()
 row.update(old_status=state[0],old_worker=state[1])
(HERE/'before-stop.json').write_text(json.dumps({'checked_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'jobs':rows},indent=2)+'\n')

def stop(row):
 cluster=row['old_worker'];result=dict(old_job=row['old_job_id'],worker=cluster,status=row['old_status'])
 if not cluster or row['old_status'] not in ('RUNNING','STARTING'):
  return dict(result,action='no active assigned attempt')
 config=json.loads((ROOT/row['profile']).read_text());run=row['run_id']
 inner='''import ray,json,time
ray.init(address='127.0.0.1:24679',namespace=RUN,log_to_driver=False)
try:
 status=ray.get_actor('runtime-status',namespace=RUN)
 state=ray.get(status.read.remote(),timeout=10)
 if state['terminal']:
  print(json.dumps({'terminal':True,'already_finished':True}))
 else:
  ray.get(status.request_stop.remote(),timeout=10)
  print(json.dumps({'stop_requested':True}),flush=True)
  deadline=time.monotonic()+300
  while time.monotonic()<deadline:
   try:
    state=ray.get(status.read.remote(),timeout=10)
   except Exception:
    print(json.dumps({'controller_departed':True}),flush=True);break
   if state['terminal']:
    print(json.dumps({'terminal':True,'exit_code':state.get('exit_code')}),flush=True);break
   time.sleep(2)
  else: raise TimeoutError('controller did not finish graceful shutdown')
finally: ray.shutdown()
'''
 script="from pathlib import Path\nimport subprocess,socket,json\n"
 script+=f"root=Path({config['root']!r}).expanduser()\n"
 script+="python=root/'envs/controller/bin/python'\n"
 script+="if not python.exists():\n print(json.dumps({'runtime_not_started':True}));raise SystemExit(0)\n"
 script+="try:\n s=socket.create_connection(('127.0.0.1',24679),timeout=3);s.close()\nexcept OSError:\n print(json.dumps({'runtime_not_listening':True}));raise SystemExit(0)\n"
 script+=f"subprocess.run([str(python),'-c',{'RUN='+repr(run)+';'+inner!r}],check=True,timeout=330)\n"
 command=['ssh','-F','/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh/'+cluster,'-o','BatchMode=yes','-o','ConnectTimeout=10',cluster,'python3 -']
 try:
  p=subprocess.run(command,input=script,text=True,capture_output=True,timeout=350)
  result.update(exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
 except subprocess.TimeoutExpired:result.update(exit=124,stderr='SSH graceful stop timed out')
 print(json.dumps({k:v for k,v in result.items() if k not in ('stdout','stderr')}),flush=True)
 return result
with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
 results=list(pool.map(stop,rows))
(HERE/'graceful-stop.json').write_text(json.dumps(results,indent=2)+'\n')
print('Graceful stop attempts recorded',flush=True)
