"""Deploy only reviewed single-host partitions; never creates or changes TPU nodes."""
import argparse,hashlib,json,os,shlex,subprocess,time
from pathlib import Path
import yaml
from launch_watch import Watch,HERE,ROOT,save
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.regional_preflight import check_qwen38,check_paths
OUT=HERE/'od-v5p64-20261010';KEY='/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/clients/7bfcb694/ssh/sky-key'
def ssh(ip,code,timeout=50):
 p=subprocess.run(['ssh','-i',KEY,'-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','ConnectTimeout=15','-o','StrictHostKeyChecking=no','-o','UserKnownHostsFile=/dev/null','-o','LogLevel=ERROR','gcpuser@'+ip,'python3 -'],input=code,text=True,capture_output=True,timeout=timeout)
 if p.returncode:raise RuntimeError(p.stdout+'\n'+p.stderr)
 return p.stdout

def main():
 ap=argparse.ArgumentParser();ap.add_argument('role');ap.add_argument('--prepare-only',action='store_true');a=ap.parse_args()
 w=Watch(False);plan=json.loads((OUT/'deployment.json').read_text());row=next(r for r in plan['rows'] if r.get('key',r['role'])==a.role)
 cfg=Config.load(ROOT/row['profile']);assert hashlib.sha256(Path(row['archive']).read_bytes()).hexdigest()==row['sha256']
 assert (OUT/'assets-ready.json').exists(),'Asset transfer not complete'
 if row['role']=='trainer':
  problem=row.get('problem','circuit')
  handoff=json.loads((OUT/('handoff-ready.json' if problem=='circuit' else problem+'-handoff-ready.json')).read_text())
  assert handoff['run_id']==cfg.run_id
  if cfg.resume_min_checkpoint_step:assert handoff['checkpoint']['batch']>=cfg.resume_min_checkpoint_step
  from tpu.swarm.ray_train.bootstrap_reuse import validate_reuse
  if cfg.bootstrap_reuse_contract_sha256:
   proof=HERE/'single-seed-east5-20261010/circuit' if problem=='circuit' else OUT
   doc=proof/('contract.json' if problem=='circuit' else problem+'-contract.json')
   summary=proof/('complete.json' if problem=='circuit' else problem+'-complete.json')
   validate_reuse(cfg,json.loads(doc.read_text()),json.loads(summary.read_text()))
  else:
   amended=json.loads((OUT/(problem+'-migrated-contract.json')).read_text())
   assert amended['contract']['config']==cfg.to_dict()
 node=json.loads(w.ops.ops.run(['gcloud','compute','tpus','tpu-vm','describe',plan['physical_node'],'--zone',plan['zone'],'--project','vision-mix','--format=json'],timeout=30))
 assert node['state']=='READY' and node.get('health')=='HEALTHY' and len(node['networkEndpoints'])==8
 assert not node.get('schedulingConfig',{}).get('preemptible')
 assert any(e['ipAddress']==row['ip'] and e['accessConfig']['externalIp']==row['external_ip'] for e in node['networkEndpoints'])
 save(OUT/(a.role+'-regional-preflight.json'),check_qwen38(cfg,w.ops.storage))
 from tpu.swarm.ray_train.build import runtime_start
 from tpu.science.package_training import science_preparation
 prep_env, prep = science_preparation(cfg)
 run = runtime_start(cfg, row['profile'])
 run = run.replace('exec python3 -m tpu.swarm.ray_train.bootstrap',
                   prep + 'exec python3 -m tpu.swarm.ray_train.bootstrap')
 audit = (ROOT/'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text()
 # Existing environments need only checkpoint/writeback headroom; cold hosts
 # retain the standard 30 GiB guard. This is the prior direct-host disk policy.
 audit = audit.replace("('home', home, 30)", "('home', home, 10 if (home / '.cache' / " + repr(row['run_id']) + "/ 'envs/controller/.complete').is_file() else 30)")
 run = "python3 - <<'SCIENCE_CLEAN_HOST'\n" + audit + "\nSCIENCE_CLEAN_HOST\n" + run
 if a.prepare_only:
  # Prepare exactly the same profile, without starting Ray, TPU engines or a client.
  run=run.rsplit('exec python3 -m tpu.swarm.ray_train.bootstrap',1)[0]+'true\n'
  row=dict(row,unit=row['unit']+'-prepare')
  a.role += '-prepare'
 env = prep_env | dict(RAY_TRAIN_CODE=row['uri'], RAY_TRAIN_CODE_SHA256=row['sha256'],
     SCIENCE_ACCELERATOR=cfg.accelerator, SKYPILOT_NODE_IPS=row['ip'], SKYPILOT_NODE_RANK='0',
     SKYPILOT_TASK_ID=row['unit'], SKYPILOT_DISABLE_LOCAL_API_SERVER='1')
 check_paths(cfg.to_dict(),'us-central1',task=env)
 script='#!/bin/bash\nset -euo pipefail\nsudo -n prlimit --pid $$ --memlock=unlimited:unlimited\n'+''.join('export '+k+'='+shlex.quote(str(v))+'\n' for k,v in env.items())+run
 (OUT/(a.role+'-startup.sh')).write_text(script)
 save(OUT/(a.role+'-startup-manifest.json'),dict(bundle_sha256=row['sha256'],script_sha256=hashlib.sha256(script.encode()).hexdigest(),profile_sha256=hashlib.sha256((ROOT/row['profile']).read_bytes()).hexdigest()))
 w.ops.upload(Path(row['archive']),row['uri'])
 # Audit before installing the script, then repeat the identical gate in the unit.
 gate=run.split("<<'SCIENCE_CLEAN_HOST'\n",1)[1].split('\nSCIENCE_CLEAN_HOST\n',1)[0]
 audit=ssh(row['external_ip'],gate);save(OUT/(a.role+'-clean-host.json'),json.loads(audit))
 code=r'''import json,subprocess,os
from pathlib import Path
r=ROW
p=Path.home()/'.local/share/skyrl-od'/r['unit'];p.mkdir(parents=True,exist_ok=True)
script=p/'launch.sh';script.write_text(SCRIPT);script.chmod(0o700)
status=subprocess.run(['systemctl','is-active',r['unit']],capture_output=True,text=True)
assert status.stdout.strip() in ('inactive','failed','unknown'),'Already active or stopping; reconcile before retry'
# The initial direct pilot used systemd-run. Its stopped transient unit
# shadows the durable service with the same name until it is retired.
transient=Path('/run/systemd/transient')/(r['unit']+'.service')
if transient.exists():
 assert subprocess.check_output(['systemctl','show',r['unit'],'--value','-p','MainPID'],text=True).strip()=='0'
 subprocess.run(['sudo','-n','mv',str(transient),str(p/'initial-transient.service')],check=True)
unit_text='[Unit]\nDescription=Isolated SkyRL on-demand '+r['role']+'\nAfter=network-online.target\nWants=network-online.target\nStartLimitIntervalSec=3600\nStartLimitBurst=3\n[Service]\nType=exec\nUser=gcpuser\nGroup=gcpuser\nWorkingDirectory=/home/gcpuser\nEnvironment=HOME=/home/gcpuser\nExecStart=/bin/bash '+str(script)+'\nRestart=on-failure\nRestartSec=60\nKillMode=control-group\nTimeoutStopSec=1000\nTasksMax=infinity\nLimitNOFILE=1048576\nLimitMEMLOCK=infinity\n[Install]\nWantedBy=multi-user.target\n'
unit_file=p/(r['unit']+'.service');unit_file.write_text(unit_text)
subprocess.run(['sudo','-n','install','-m','644',str(unit_file),'/etc/systemd/system/'+unit_file.name],check=True)
subprocess.run(['sudo','-n','systemctl','daemon-reload'],check=True)
subprocess.run(['sudo','-n','systemctl','reset-failed',r['unit']],capture_output=True)
action=['start'] if PREPARE_ONLY else ['enable','--now']
p=subprocess.run(['sudo','-n','systemctl',*action,r['unit']],capture_output=True,text=True)
print(json.dumps(dict(rc=p.returncode,stdout=p.stdout,stderr=p.stderr)));p.check_returncode()
'''.replace('ROW',repr(row)).replace('SCRIPT',repr(script)).replace('PREPARE_ONLY',repr(a.prepare_only))
 save(OUT/(a.role+'-launch-intent.json'),dict(time=time.time(),row=row));result=ssh(row['external_ip'],code)
 save(OUT/(a.role+'-launched.json'),dict(time=time.time(),result=json.loads(result),row=row));print(result)
if __name__=='__main__':main()
