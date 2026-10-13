import json,time,concurrent.futures
from launch_watch import save
from od_deploy import OUT,ssh
REMOTE=r'''import json,subprocess,shutil,urllib.request
from pathlib import Path
row=ROW
run=Path.home()/'.cache'/row['run_id']/'runs'/row['run_id']
out={'run_id':row['run_id'],'role':row['role'],'disk_free_gib':round(shutil.disk_usage(Path.home()).free/2**30,1),'logs':{}}
out['unit']=subprocess.run(['systemctl','show',row['unit'],'-p','ActiveState','-p','SubState','-p','NRestarts','-p','ExecMainStatus'],capture_output=True,text=True).stdout
for name in ['bootstrap-0.jsonl','host-0.jsonl','controller.jsonl','driver.log','trainer.log','client.log']:
 p=run/name
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-12000));text=f.read().decode(errors='replace')
  out['logs'][name]=text.splitlines()[-5:]
for p in sorted(run.glob('systemd-*/service.log'))[-1:]:
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-8000));text=f.read().decode(errors='replace')
 out['logs']['service.log']=text.splitlines()[-8:]
try:
 with urllib.request.urlopen('http://127.0.0.1:24800/status',timeout=5) as r:s=json.load(r)
 out['serving']={k:s.get(k) for k in ['state','ready_engines','expected_engines','active','owner_run','fatal_error','borrowing','capabilities']}
 if out['serving'].get('capabilities'):out['serving']['capabilities'].pop('contract',None)
except Exception as e:out['serving_error']=str(e)
cp=run/'client/tinker_log'/row['run_id']/'member_qwen/checkpoints.jsonl'
if cp.exists():out['checkpoint']=json.loads(cp.read_text().splitlines()[-1])
print(json.dumps(out))
'''
def main():
 plan=json.loads((OUT/'deployment.json').read_text())
 def check(row):
  try:return json.loads(ssh(row['external_ip'],REMOTE.replace('ROW',repr(row))))
  except Exception as e:return dict(role=row['role'],error=str(e))
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:rows=list(ex.map(check,plan['rows']))
 save(OUT/'live.json',dict(time=time.time(),rows=rows))
 for row in rows:
  print(json.dumps({k:row.get(k) for k in ['role','run_id','unit','disk_free_gib','serving','serving_error','checkpoint','error']}),flush=True)
  print(json.dumps({'latest':row.get('logs',{}).get('driver.log',[])[-2:]}),flush=True)
if __name__=='__main__':main()
