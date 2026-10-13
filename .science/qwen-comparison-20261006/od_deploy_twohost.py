"""Deploy a validated pair atomically after handoff; leave all other hosts alone."""
import argparse,hashlib,json,shlex,time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from od_deploy import ssh,OUT
from launch_watch import Watch,ROOT,save
from tpu.swarm.ray_train.config import Config
from tpu.swarm.ray_train.regional_preflight import check_qwen38,check_paths
from tpu.swarm.ray_train.assigned_topology import verify_partition
from tpu.swarm.ray_train.build import runtime_start
from tpu.science.package_training import science_preparation
ap=argparse.ArgumentParser();ap.add_argument('problem');a=ap.parse_args()
r=next(r['new'] for r in json.loads((OUT/'twohost-bundles.json').read_text()) if r['new']['problem']==a.problem)
cfg=Config.load(ROOT/r['profile']);cfg.validate();assert hashlib.sha256(Path(r['archive']).read_bytes()).hexdigest()==r['sha256']
proof=json.loads((OUT/f"two-host-probe-{'-'.join(map(str,r['physical_ranks']))}-{cfg.trainer.process_bounds}.json").read_text());assert all(x['returncode']==0 for x in proof)
verify_partition([x['records'][0] for x in proof],cfg.trainer.process_bounds,cfg.trainer.chip_bounds)
handoff=json.loads((OUT/(a.problem+'-twohost-handoff-ready.json')).read_text());assert handoff['run_id']==r['run_id'] and handoff['checkpoint']['batch']>=cfg.resume_min_checkpoint_step
w=Watch(False);node=json.loads(w.ops.ops.run(['gcloud','compute','tpus','tpu-vm','describe',cfg.physical_node,'--zone',cfg.zone,'--project','vision-mix','--format=json'],timeout=30));assert node['state']=='READY' and node['health']=='HEALTHY'
for ip,external in zip(r['ips'],r['external_ips']):assert any(e['ipAddress']==ip and e['accessConfig']['externalIp']==external for e in node['networkEndpoints'])
save(OUT/(a.problem+'-twohost-regional-preflight.json'),check_qwen38(cfg,w.ops.storage));w.ops.upload(Path(r['archive']),r['uri'])
prep_env,prep=science_preparation(cfg);run=runtime_start(cfg,r['profile']).replace('exec python3 -m tpu.swarm.ray_train.bootstrap',prep+'exec python3 -m tpu.swarm.ray_train.bootstrap')
audit=(ROOT/'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text().replace("('home', home, 30)","('home', home, 10 if (home / '.cache' / "+repr(r['run_id'])+"/ 'envs/controller/.complete').is_file() else 30)")
for i,ip in enumerate(r['external_ips']):save(OUT/f'{a.problem}-twohost-clean-{i}.json',json.loads(ssh(ip,audit)))
scripts=[]
for rank in range(2):
 env=prep_env|dict(RAY_TRAIN_CODE=r['uri'],RAY_TRAIN_CODE_SHA256=r['sha256'],SCIENCE_ACCELERATOR=cfg.accelerator,SKYPILOT_NODE_IPS=' '.join(r['ips']),SKYPILOT_NODE_RANK=str(rank),SKYPILOT_TASK_ID=r['unit']+'-twohost-20261013',SKYPILOT_DISABLE_LOCAL_API_SERVER='1')
 check_paths(cfg.to_dict(),'us-central1',task=env)
 script='#!/bin/bash\nset -euo pipefail\nsudo -n prlimit --pid $$ --memlock=unlimited:unlimited\n'+''.join('export '+k+'='+shlex.quote(str(v))+'\n' for k,v in env.items())+"python3 - <<'SCIENCE_CLEAN_HOST'\n"+audit+"\nSCIENCE_CLEAN_HOST\n"+run
 scripts.append(script);(OUT/f'{a.problem}-twohost-startup-{rank}.sh').write_text(script)
install=r'''import json,subprocess
from pathlib import Path
unit=UNIT;p=Path.home()/'.local/share/skyrl-od'/unit;p.mkdir(parents=True,exist_ok=True)
assert subprocess.check_output(['systemctl','show',unit,'--value','-p','MainPID'],text=True).strip() in ('0','')
assert subprocess.run(['systemctl','is-active',unit],capture_output=True,text=True).stdout.strip() in ('inactive','failed','unknown','')
script=p/'launch.sh'
if script.exists():(p/'launch-before-twohost.sh').write_bytes(script.read_bytes())
script.write_text(SCRIPT);script.chmod(0o700)
transient=Path('/run/systemd/transient')/(unit+'.service')
if transient.exists():subprocess.run(['sudo','-n','mv',str(transient),str(p/'initial-transient.service')],check=True)
text='[Unit]\nDescription=Two-host SkyRL trainer\nAfter=network-online.target\nWants=network-online.target\nStartLimitIntervalSec=3600\nStartLimitBurst=3\n[Service]\nType=exec\nUser=gcpuser\nGroup=gcpuser\nWorkingDirectory=/home/gcpuser\nEnvironment=HOME=/home/gcpuser\nExecStart=/bin/bash '+str(script)+'\nRestart=on-failure\nRestartSec=60\nKillMode=control-group\nTimeoutStopSec=1000\nTasksMax=infinity\nLimitNOFILE=1048576\nLimitMEMLOCK=infinity\n[Install]\nWantedBy=multi-user.target\n'
f=p/(unit+'.service');f.write_text(text);subprocess.run(['sudo','-n','install','-m','644',str(f),'/etc/systemd/system/'+f.name],check=True);subprocess.run(['sudo','-n','rm','-f',str(Path('/run/systemd/system')/(unit+'.service.d')/'handoff.conf')],check=True);subprocess.run(['sudo','-n','systemctl','daemon-reload'],check=True);subprocess.run(['sudo','-n','systemctl','reset-failed',unit],capture_output=True);print('installed')
'''
for ip,script in zip(r['external_ips'],scripts):print(ip,ssh(ip,install.replace('UNIT',repr(r['unit'])).replace('SCRIPT',repr(script))))
save(OUT/(a.problem+'-twohost-launch-intent.json'),dict(time=time.time(),row=r))
start="import subprocess;subprocess.run(['sudo','-n','systemctl','enable','--now',"+repr(r['unit'])+"],check=True);print('started')"
with ThreadPoolExecutor(2) as ex:results=list(ex.map(lambda ip:ssh(ip,start,timeout=45),r['external_ips']))
save(OUT/(a.problem+'-twohost-launched.json'),dict(time=time.time(),row=r,results=results))
plan=json.loads((OUT/'deployment.json').read_text());save(OUT/(a.problem+'-deployment-before-twohost.json'),plan);plan['rows']=[r if x['run_id']==r['run_id'] else x for x in plan['rows']];save(OUT/'deployment.json',plan)
discovery=json.loads((OUT/'discovery-targets.json').read_text());discovery['trainers']=[x for x in discovery['trainers'] if 'erdos' not in x['run_id'] and 'qubit' not in x['run_id']]
for x in discovery['trainers']:
 if x['run_id']==r['run_id']:x.update(external_ip=r['external_ip'],stable_urls=cfg.inference.external_pool_urls[cfg.model],allow_spot=True,enable_spot_after_stable_completion=False)
save(OUT/'discovery-targets.json',discovery);print(a.problem,results)
