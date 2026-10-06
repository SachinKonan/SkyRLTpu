"""Durable Gemma -> Muse -> Qwen queue on the one authorized standalone slice.

Uses the same packaged Ray v2 task, with coordinated all-host retries. Never
stops units outside this queue, changes machine allocation, or skips a model.
"""
import argparse,concurrent.futures,datetime,fcntl,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
import yaml
from google.cloud import storage
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).parent
MACHINE='shu-v5p-64-on-demand-machine1';ZONE='us-central1-a'
KEY='/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/clients/7bfcb694/ssh/sky-key'

def write(path,value):
 temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2)+'\n');temp.replace(path)
def ssh(ip,code):
 p=subprocess.run(['ssh','-i',KEY,'-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=10','gcpuser@'+ip,'python3 -'],input=code,text=True,capture_output=True,timeout=75)
 return dict(exit=p.returncode,stdout=p.stdout[-12000:],stderr=p.stderr[-1500:])
def parallel(ips,fn):
 with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:return list(ex.map(fn,enumerate(ips)))
def machine():
 p=subprocess.run([os.environ.get('GCLOUD_BIN','gcloud'),'compute','tpus','tpu-vm','describe',MACHINE,'--zone',ZONE,'--project','vision-mix','--format=json'],capture_output=True,text=True,check=True,timeout=60)
 d=json.loads(p.stdout);assert d['state']=='READY' and d['acceleratorType']=='v5p-64'
 endpoints=d['networkEndpoints'];assert len(endpoints)==8
 return [x['ipAddress'] for x in endpoints],[x['accessConfig']['externalIp'] for x in endpoints]
def completed(client,row):
 b=client.bucket(row['bucket'][5:]);prefix='ray-training/'+row['run_id']+'/client/tinker_log/'+row['run_id']+'/'
 try:
  metrics=[json.loads(x) for x in b.blob(prefix+'metrics.jsonl').download_as_text().splitlines() if x.strip()]
  cp=[json.loads(x) for x in b.blob(prefix+'member_'+row['model']+'/checkpoints.jsonl').download_as_text().splitlines() if x.strip()]
 except Exception:return False
 return any(m.get('step')==10 and not any(v for k,v in m.items() if 'train_error' in k or 'consec_all_member_errors' in k) for m in metrics) and any(r.get('batch')==10 for r in cp)
def start(row,attempt,internal,external):
 task=yaml.safe_load((ROOT/row['task']).read_text());assert hashlib.sha256((ROOT/row['task']).read_bytes()).hexdigest()==row['task_sha256']
 audit=(ROOT/'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text()
 checks=parallel(external,lambda pair:dict(rank=pair[0],**ssh(pair[1],audit)))
 write(HERE/(row['label']+f'-audit-{attempt}.json'),checks)
 if not all(r['exit']==0 for r in checks):raise RuntimeError('clean-host audit failed; no launch')
 unit='skyrl-circuit10-'+row['model']+'-a'+str(attempt)
 common='set -euo pipefail\n'+'\n'.join('export '+k+'='+shlex.quote(str(v)) for k,v in task['envs'].items())+'\n'
 common+='export SKYPILOT_NODE_IPS='+shlex.quote('\n'.join(internal))+'\nexport SKYPILOT_TASK_ID='+shlex.quote(row['label']+'-a'+str(attempt))+'\n'
 common+='python3 - <<\'HOST_AUDIT\'\n'+audit+'\nHOST_AUDIT\n'+task['run']
 receipt=dict(unit=unit,run_id=row['run_id'],attempt=attempt,state='dispatching',internal_ips=internal,external_ips=external,started=time.time())
 write(HERE/(row['label']+'-launch.json'),receipt)
 def launch(pair):
  rank,ip=pair;script='/home/gcpuser/.cache/'+row['run_id']+'/circuit10-launch.sh';log=script+'.log'
  body='export SKYPILOT_NODE_RANK='+str(rank)+'\n'+common
  command=['sudo','-n','systemd-run','--quiet','--unit='+unit,'--uid=gcpuser','--gid=gcpuser','--property=Type=exec','--property=KillMode=control-group','--property=TimeoutStopSec=120','--property=TasksMax=infinity','--property=LimitNOFILE=1048576','--property=LimitMEMLOCK=infinity','--property=StandardOutput=append:'+log,'--property=StandardError=append:'+log,'--working-directory=/home/gcpuser','/bin/bash',script]
  code='from pathlib import Path\nimport subprocess\np=Path('+repr(script)+');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('+repr(body)+');p.chmod(0o700)\nsubprocess.run('+repr(command)+',check=True)\n'
  return dict(rank=rank,**ssh(ip,code))
 results=parallel(external,launch);receipt.update(hosts=results,state='launched' if all(x['exit']==0 for x in results) else 'partial')
 write(HERE/(row['label']+'-launch.json'),receipt)
 return receipt

def tick(client,rows):
 state_path=HERE/'circuit-queue-state.json';state=json.loads(state_path.read_text()) if state_path.exists() else dict(index=0,attempt=0,phase='queued')
 if state['index']>=len(rows) or state['phase']=='blocked':return state
 row=rows[state['index']]
 if state.get('unit'):
  external=state['external_ips'];unit=state['unit']
  code='import subprocess,json\np=subprocess.run(["systemctl","show",'+repr(unit)+',"-p","ActiveState","-p","SubState","-p","Result"],capture_output=True,text=True)\nprint(json.dumps(dict(line.split("=",1) for line in p.stdout.splitlines() if "=" in line)))\n'
  checks=parallel(external,lambda pair:ssh(pair[1],code))
  if any(r['exit'] for r in checks):return dict(state,observation='host unreachable')
  states=[json.loads(r['stdout']) for r in checks]
  active=[s['ActiveState'] in ('active','activating','deactivating') for s in states]
  if all(active):return state
  if completed(client,row):
   if any(active):return dict(state,observation='completed; waiting for host cleanup')
   state=dict(index=state['index']+1,attempt=0,phase='queued');write(state_path,state);return state
  # Any dead host aborts this gang. Stop only this attempt's exact outer unit.
  stopped=parallel(external,lambda pair:ssh(pair[1],'import subprocess\nsubprocess.run(["sudo","-n","systemctl","stop",'+repr(unit)+'],check=True,timeout=60)\n'))
  if any(r['exit'] for r in stopped):raise RuntimeError('failed to stop own failed gang')
  state.pop('unit',None);state['phase']='retry' if state['attempt']<3 else 'blocked';write(state_path,state)
  return state
 internal,external=machine();attempt=state['attempt']+1
 # Persist intent before remote dispatch: restart reconciles a partial attempt.
 unit='skyrl-circuit10-'+row['model']+'-a'+str(attempt)
 state.update(attempt=attempt,phase='launching',unit=unit,external_ips=external,run_id=row['run_id']);write(state_path,state)
 try:
  receipt=start(row,attempt,internal,external)
 except Exception as exc:
  state.update(phase='blocked',error=str(exc)[:500]);write(state_path,state)
  raise
 state['phase']=receipt['state'];write(state_path,state);return state

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--once',action='store_true');args=parser.parse_args()
 lock=(HERE/'circuit-queue.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 client=storage.Client(project='vision-mix');assert client._credentials.service_account_email=='289186856710-compute@developer.gserviceaccount.com'
 manifest=json.loads((HERE/'prepared.json').read_text());rows=[next(r for r in manifest['jobs'] if r['kind']=='circuit' and r['model']==m) for m in ['gemma','muse','qwen']]
 receipts=json.loads((HERE/'uploads.json').read_text());uploaded={r['uri']:r['sha256'] for r in receipts}
 for row in rows:assert uploaded[row['code_uri']]==row['archive_sha256']
 while True:
  try:state=tick(client,rows);print(json.dumps(dict(time=time.time(),**state)),flush=True)
  except Exception as e:print(json.dumps(dict(time=time.time(),error=type(e).__name__,detail=str(e)[:500])),flush=True)
  if args.once:return
  time.sleep(30)
if __name__=='__main__':main()
