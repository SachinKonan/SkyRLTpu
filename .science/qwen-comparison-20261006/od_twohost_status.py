"""Read-only status for the two paired trainers; never print lease credentials."""
import json,time
from concurrent.futures import ThreadPoolExecutor
from od_deploy import ssh,OUT
from launch_watch import save
REMOTE=r'''from pathlib import Path
import json,subprocess,urllib.request,shutil
r=ROW;run=Path.home()/'.cache'/r['run_id']/'runs'/r['run_id'];out={'problem':r['problem'],'physical_hosts':r['physical_ranks'],'target':15,'unit':subprocess.check_output(['systemctl','show',r['unit'],'-p','ActiveState','-p','NRestarts'],text=True),'free_gib':round(shutil.disk_usage(Path.home()).free/2**30,1)}
def tail(p,cap=300000):
 if not p.exists():return ''
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-cap));return f.read().decode(errors='replace')
rows=[]
for l in tail(run/'controller.jsonl').splitlines():
 try:rows.append(json.loads(l))
 except ValueError:pass
hosts=[x for x in rows if x.get('event')=='hosts']
if hosts:out['hosts_time']=hosts[-1]['time'];out['hosts']=hosts[-1]['statuses']
failed=[x for x in rows if x.get('event')=='controller_failed']
if failed:out['last_failure']={k:failed[-1][k] for k in ('time','detail')}
cp=run/'client/tinker_log'/r['run_id']/'member_qwen/checkpoints.jsonl'
if cp.exists():out['checkpoint']=json.loads(cp.read_text().splitlines()[-1])
text=tail(run/'client.log',2500000);out['resume']=[l for l in text.splitlines() if any(s in l for s in ['Resume optimizer','member qwen resumed','resuming shared PUCT'])][-3:]
try:
 d=json.load(urllib.request.urlopen('http://127.0.0.1:24800/status',timeout=5));b=d.get('borrowing') or {};out['farms']={k:b.get(k) for k in ('ready','held','engines','active','target','candidates')};out['active']=d.get('active')
except Exception as e:out['farm_error']=str(e)
print(json.dumps(out))'''
rows=[r for r in json.loads((OUT/'deployment.json').read_text())['rows'] if r['role']=='trainer' and r.get('problem') in ('circuit','ac1')]
with ThreadPoolExecutor(2) as ex:results=list(ex.map(lambda r:json.loads(ssh(r['external_ip'],REMOTE.replace('ROW',repr(r)),timeout=30)),rows))
save(OUT/'twohost-live.json',{'time':time.time(),'rows':results})
for r in results:
 print(json.dumps({k:r.get(k) for k in ('problem','physical_hosts','unit','checkpoint','resume','farms','farm_error','free_gib')},sort_keys=True))
 print('host_phases',[(h['rank'],h['phase'],h['processes'].get('trainer')) for h in r.get('hosts',[])])
 if r.get('last_failure') and r['last_failure']['time']>time.time()-120:print('recent failure',r['last_failure'])
