"""Read-only inventory of this experiment on the four explicitly assigned hosts."""
import concurrent.futures,datetime,json,shlex,subprocess
from pathlib import Path
CLUSTER='tpuswarm-v5p32-east5a-erdos-3242'
CONF='/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh/'+CLUSTER
IPS=['10.202.15.214','10.202.15.216','10.202.15.213','10.202.15.215']
script='''import json
from pathlib import Path
root=Path('/dev/shm/fair-ae-research');out={}
for p in root.iterdir():
 if p.is_dir() and (p/'config.json').exists():
  d={}
  for name in ['config.json','best.json','complete.json','density/report.json','development_diagnostics.json','linear_probe.json','snapshot.json']:
   q=p/name
   if q.exists():
    try:d[name]=json.loads(q.read_text())
    except Exception as e:d[name]={'read_error':str(e)}
  out[p.name]=d
print(json.dumps(out))
'''
def fetch(i):
 cmd='python3 -c '+shlex.quote(script)
 if i:cmd='ssh -n -c aes128-gcm@openssh.com -F /dev/null -i ~/.ssh/sky-cluster-key -o StrictHostKeyChecking=no -o ConnectTimeout=15 '+IPS[i]+' '+shlex.quote(cmd)
 r=subprocess.run(['ssh','-c','aes128-gcm@openssh.com','-F',CONF,'-o','BatchMode=yes','-o','ConnectTimeout=15',CLUSTER,cmd],capture_output=True,text=True,timeout=45)
 if r.returncode:return str(i),{'error':r.stderr}
 return str(i),json.loads(r.stdout)
with concurrent.futures.ThreadPoolExecutor(4) as e:hosts=dict(e.map(fetch,range(4)))
result={'collected_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cluster':CLUSTER,'hosts':hosts}
path=Path('.fair/ae-research-evidence/status.json');path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2))
for rank,runs in hosts.items():
 for name,d in runs.items():
  if isinstance(d,dict):
   b=d.get('best.json',{});den=d.get('density/report.json',{});dev=d.get('development_diagnostics.json',{})
   print(json.dumps(dict(host=rank,name=name,best_step=b.get('step'),rec=b.get('reconstruction_nmse',b.get('clean_summary_nmse')),params=b.get('parameter_rmse'),complete='complete.json' in d,k=den.get('selected_k'),diagnosed=bool(dev))))
