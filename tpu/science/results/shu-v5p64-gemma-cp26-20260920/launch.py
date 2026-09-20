"""Launch the existing Ray v2 runtime on the explicitly authorized on-demand slice."""
import concurrent.futures,datetime,hashlib,json,shlex,subprocess
from pathlib import Path
import yaml
from google.cloud import storage
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
run='shu-v5p64-gemma-cp26-grpo-lr4e5-s1-20260920';folder=ROOT/'.science/packages/shu-v5p64-gemma-cp26-20260920'
assert not (HERE/'launch-receipt.json').exists(),'Reconcile the previous launch before retrying'
assert all(r['exit']==0 for r in json.loads((HERE/'host-audit.json').read_text()))
task=yaml.safe_load((folder/(run+'.yaml')).read_text());seed=json.loads((HERE/'seed-import.json').read_text())
c=storage.Client(project='vision-mix');assert c._credentials.service_account_email=='289186856710-compute@developer.gserviceaccount.com'
def upload(uri,path):
 b,n=uri[5:].split('/',1);blob=c.bucket(b).blob(n)
 if not blob.exists():blob.upload_from_filename(path,if_generation_match=0,timeout=180)
 assert hashlib.sha256(blob.download_as_bytes(timeout=180)).hexdigest()==hashlib.sha256(path.read_bytes()).hexdigest()
upload(task['envs']['RAY_TRAIN_CODE'],folder/'ray-training.tar.gz')
assert not next(iter(c.list_blobs('sk7524-tinker-tpu-us-central1',prefix='ray-training/'+run+'/',max_results=1)),None),'Run namespace already exists'
upload(seed['destination'],HERE/'seed-pool.json')
internal=['10.128.1.94','10.128.1.95','10.128.1.91','10.128.1.93','10.128.1.88','10.128.1.90','10.128.1.89','10.128.1.92']
external=['34.58.61.190','34.136.9.17','34.122.223.241','35.222.180.160','35.223.208.106','34.31.132.165','35.239.3.84','34.134.8.42']
unit='skyrl-shu-gemma-cp26-launch';token=run+'-attempt1'
audit=(ROOT/'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text()
common='set -euo pipefail\n'+'\n'.join('export '+k+'='+shlex.quote(v) for k,v in task['envs'].items())+'\n'
common+='export SKYPILOT_NODE_IPS='+shlex.quote('\n'.join(internal))+'\nexport SKYPILOT_TASK_ID='+shlex.quote(token)+'\n'
common+="python3 - <<'HOST_AUDIT'\n"+audit+'\nHOST_AUDIT\n'+task['run']
record={'run_id':run,'machine':'shu-v5p-64-on-demand-machine1','zone':'us-central1-a','unit':unit,'state':'dispatching','time':datetime.datetime.now(datetime.timezone.utc).isoformat(),'code_uri':task['envs']['RAY_TRAIN_CODE'],'internal_ips':internal,'external_ips':external}
(HERE/'launch-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
def launch(pair):
 rank,ip=pair
 body='export SKYPILOT_NODE_RANK='+str(rank)+'\n'+common
 remote='/home/gcpuser/.cache/'+run+'/launch.sh';log='/home/gcpuser/.cache/'+run+'/launcher.log'
 code='import os,subprocess\nfrom pathlib import Path\np=Path('+repr(remote)+');p.parent.mkdir(parents=True,exist_ok=True);p.write_text('+repr(body)+');p.chmod(0o700)\n'
 args=['sudo','-n','systemd-run','--quiet','--unit='+unit,'--uid=gcpuser','--gid=gcpuser','--property=Type=exec','--property=KillMode=control-group','--property=TimeoutStopSec=120','--property=TasksMax=infinity','--property=LimitNOFILE=1048576','--property=LimitMEMLOCK=infinity','--property=StandardOutput=append:'+log,'--property=StandardError=append:'+log,'--working-directory=/home/gcpuser','/bin/bash',remote]
 code+='subprocess.run('+repr(args)+',check=True)\nsubprocess.run('+repr(['systemctl','show',unit,'-p','ActiveState','-p','MainPID'])+',check=True)\n'
 p=subprocess.run(['ssh','-i','/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/clients/7bfcb694/ssh/sky-key','-o','BatchMode=yes','-o','IdentitiesOnly=yes','-o','ConnectTimeout=8','gcpuser@'+ip,'python3 -'],input=code,capture_output=True,text=True,timeout=35)
 return dict(rank=rank,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:result=list(ex.map(launch,enumerate(external)))
record.update(hosts=result,state='launched' if all(r['exit']==0 for r in result) else 'partial-dispatch-needs-reconciliation')
(HERE/'launch-receipt.json').write_text(json.dumps(record,indent=2)+'\n')
for r in result:print(r,flush=True)
assert record['state']=='launched'
