import collections,concurrent.futures,datetime,fcntl,json,os,sqlite3,subprocess,time
from pathlib import Path
ROOT=Path('/scratch/gpfs/ZHUANGL/sk7524/SkyRLTpu-science-placement')
OUT=ROOT/'.science/monitor-20260920'
OUT.mkdir(parents=True,exist_ok=True)
lock=(OUT/'watch.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
(OUT/'watch.pid').write_text(str(os.getpid()))
manifest=json.loads((ROOT/'tpu/science/results/serve-handoff-fix-20260920/submissions.json').read_text())['jobs']
meta={r['job_id']:r for r in manifest}
for r in json.loads((ROOT/'tpu/science/results/capacity-seven-20260920/submissions.json').read_text())['jobs']: meta[r['job_id']]=r
meta['shu-v5p64']={'run_id':'shu-v5p64-gemma-cp26-grpo-lr4e5-s1-20260920','model':'gemma','task':'cp26','direct_host':'34.58.61.190'}
meta[1270]={'run_id':'science-circuit-v6e-qwen-borrow173-grpo-20260919-fix1','model':'qwen','task':'circuit-borrowing'}
REMOTE=r'''
import collections,json,time
from pathlib import Path
root=Path('/home/gcpuser/.cache')/RUN/'runs'/RUN
out={'root_exists':root.exists(),'files':{}}
def tail(p,limit=1600000):
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-limit));return f.read().decode(errors='replace').splitlines()
p=root/'controller.jsonl'
if p.exists():
 out['controller_age_s']=round(time.time()-p.stat().st_mtime,1)
 rows=[]
 for line in tail(p):
  try: rows.append(json.loads(line))
  except Exception:pass
 if rows:
  pid=rows[-1].get('pid');rows=[r for r in rows if r.get('pid')==pid]
  out['controller_pid']=pid;out['last_event']=rows[-1].get('event')
  out['events']=[{k:v for k,v in r.items() if k not in ('host','run_id','pid')} for r in rows if r.get('event') not in ('hosts','inference_status','writeback_complete','cache_writeback_complete','heartbeat')][-12:]
  h=next((r for r in reversed(rows) if r.get('event')=='hosts'),None)
  if h:out['hosts']=[{k:v for k,v in h1.items() if k in ('rank','role','phase','cache_sync_error','stopped','processes')} for h1 in h['statuses']]
  s=next((r for r in reversed(rows) if r.get('event')=='inference_status'),None)
  if s:out['inference']={k:v for k,v in s.items() if k in ('expected','replicas','exhausted','starts','version')}
for name in ('driver.log','bootstrap.log','client.log'):
 p=root/name
 if p.exists():out['files'][name]={'age_s':round(time.time()-p.stat().st_mtime,1),'bytes':p.stat().st_size,'tail':tail(p,12000)[-5:]}
b=root/'client'/'bootstrap'
if b.exists():
 grades=list(b.glob('drafts/group-*/grade-*.json'));valid=0;best=None
 for p in grades:
  try:r=json.loads(p.read_text());valid+=r.get('correctness')==1;best=max(best if best is not None else float('-inf'),r.get('reward',0))
  except Exception:pass
 out['bootstrap']={'generated_groups':len(list(b.glob('drafts/group-*/generation.json'))),'graded':len(grades),'valid':valid,'best_reward':best}
 if (b/'complete.json').exists():out['bootstrap']['complete']=json.loads((b/'complete.json').read_text())
logs=root/'client'/'tinker_log'/RUN
out['pools']=[p.name for p in sorted(logs.glob('puct_sampler_step_*.json'))][-3:]
for p in logs.glob('member_*/checkpoints.jsonl'):
 out.setdefault('checkpoints',{})[str(p.relative_to(logs))]=tail(p,8000)[-2:]
for p in logs.glob('**/metrics.jsonl'):
 out.setdefault('metrics',{})[str(p.relative_to(logs))]=tail(p,16000)[-1:]
p=root.parents[1]/'launcher.log'
if p.exists():
 out['launcher_log_tail']=tail(p,6000)[-5:]
 import subprocess
 status=subprocess.run(['systemctl','show','skyrl-shu-gemma-cp26-launch','-p','ActiveState','-p','ExecMainStatus'],capture_output=True,text=True,timeout=5)
 out['launcher_service']=status.stdout
print(json.dumps(out))
'''
def probe(row):
 if not row['cluster'] or row['status'] not in ('STARTING','RUNNING','RECOVERING'):return row
 cluster=row['cluster'];conf=Path('/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh')/cluster
 if not conf.exists() and not row.get('direct_host'):return dict(row,probe_error='No SSH configuration')
 script='RUN='+repr(row['run_id'])+'\n'+REMOTE
 try:
  ssh=(['ssh','-i','/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/clients/7bfcb694/ssh/sky-key','-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=8','gcpuser@'+row['direct_host']] if row.get('direct_host') else ['ssh','-F',str(conf),'-o','BatchMode=yes','-o','ConnectTimeout=8',cluster])
  p=subprocess.run(ssh+['python3 -'],input=script,text=True,capture_output=True,timeout=35)
  row['runtime']=json.loads(p.stdout) if p.returncode==0 else {'probe_error':p.stderr[-700:]}
  if row.get('direct_host'):
   service=row['runtime'].get('launcher_service','')
   row['status']='FAILED' if 'ActiveState=failed' in service else ('RUNNING' if 'ActiveState=active' in service else 'UNKNOWN')
 except Exception as e:row['probe_error']=type(e).__name__+': '+str(e)[:300]
 return row
while True:
 started=time.monotonic();stamp=datetime.datetime.now(datetime.timezone.utc).isoformat()
 try:
  c=sqlite3.connect('file:/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/spot_jobs.db?mode=ro',uri=True)
  rows=[]
  for jid,m in meta.items():
   if m.get('direct_host'):
    rows.append(dict(job_id=jid,run_id=m['run_id'],model=m['model'],task=m['task'],pool='v5p64-on-demand-central1a',status='RUNNING',cluster='shu-v5p-64-on-demand-machine1',direct_host=m['direct_host']))
    continue
   r=c.execute('select j.pool,s.status,j.current_cluster_name from job_info j join spot s on j.spot_job_id=s.spot_job_id where j.spot_job_id=?',(jid,)).fetchone()
   if r:rows.append(dict(job_id=jid,run_id=m['run_id'],model=m['model'],task=m['task'],pool=r[0],status=r[1],cluster=r[2]))
  c.close()
  with concurrent.futures.ThreadPoolExecutor(max_workers=12) as ex:rows=list(ex.map(probe,rows))
  snap={'checked_at':stamp,'duration_s':round(time.monotonic()-started,1),'jobs':rows}
  tmp=OUT/'latest.tmp';tmp.write_text(json.dumps(snap,indent=2)+'\n');tmp.replace(OUT/'latest.json')
  with (OUT/'history.jsonl').open('a') as f:f.write(json.dumps(snap)+'\n')
  print(stamp,flush=True)
  for pool in sorted({r['pool'] for r in rows}):
   rr=[r for r in rows if r['pool']==pool];print(pool,dict(collections.Counter(r['status'] for r in rr)),'assigned',sum(bool(r['cluster']) for r in rr),flush=True)
  for r in rows:
   rt=r.get('runtime',{});print(r['job_id'],r['model'],r['task'],r['status'],'controller_age',rt.get('controller_age_s'),'event',rt.get('last_event'),'bootstrap',rt.get('bootstrap'),'pools',rt.get('pools'),flush=True)
  if all(r['status'] in ('SUCCEEDED','FAILED','FAILED_SETUP','CANCELLED') for r in rows):break
 except Exception as e:print('monitor_error',type(e).__name__,str(e),flush=True)
 time.sleep(max(1,240-(time.monotonic()-started)))
