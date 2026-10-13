"""Durably snapshot a stopped trainer without changing its scientific state."""
import argparse,json,time
from od_deploy import ssh,OUT
from launch_watch import Watch,save
ap=argparse.ArgumentParser();ap.add_argument('problem');a=ap.parse_args();w=Watch(False)
r=next(x for x in json.loads((OUT/'deployment.json').read_text())['rows'] if x['role']=='trainer' and x.get('problem','circuit')==a.problem)
code=r'''import subprocess,json,os,hashlib,base64
from pathlib import Path
r=ROW;root=Path.home()/'.cache'/r['run_id'];run=root/'runs'/r['run_id'];client=run/'client/tinker_log'/r['run_id'];prefix='gs://sk7524-tinker-tpu-us-central1/ray-training/'+r['run_id']
assert subprocess.check_output(['systemctl','show',r['unit'],'--value','-p','MainPID'],text=True).strip()=='0'
cp=json.loads((client/'member_qwen/checkpoints.jsonl').read_text().splitlines()[-1]);step=cp['batch'];snap=client/f'puct_sampler_step_{step:06d}.json';assert snap.is_file()
code=Path.home()/'.cache/skyrl-ray-code'/r['sha256'];env=dict(os.environ,PYTHONPATH=str(code))
if REUSE_BACKUP:
 assert (run/'tinker-backup.db').is_file()
 assert 'final_run_writeback_complete' in (run/'controller.jsonl').read_text()[-5000:]
else:subprocess.run([str(root/'envs/controller/bin/python'),'-c',"from pathlib import Path;from tpu.swarm.ray_train.database_snapshot import create_snapshot;create_snapshot(Path("+repr(str(run/'tinker.db'))+"),Path("+repr(str(run/'tinker-backup.db'))+"))"],env=env,check=True)
gcloud='/snap/google-cloud-cli/current/bin/gcloud'
for args in [['cp','--recursive',str(run/'client'),prefix+'/'],['cp',str(run/'tinker-backup.db'),prefix+'/tinker-backup.db']]:
 p=subprocess.run([gcloud,'storage',*args],capture_output=True,text=True,timeout=300)
 if p.returncode:raise RuntimeError(p.stderr[-3000:])
checks={};model=cp['state_path'].split('/')[2]
for relative in [f'checkpoints/{model}/{step:06d}.tar.gz',f'checkpoints/{model}/sampler_weights/{step:06d}.tar.gz']:
 f=run/relative;h=hashlib.md5()
 with f.open('rb') as stream:
  while b:=stream.read(8*2**20):h.update(b)
 checks[relative]={'size':f.stat().st_size,'md5':base64.b64encode(h.digest()).decode()}
print(json.dumps({'run_id':r['run_id'],'checkpoint':cp,'snapshot_sha256':hashlib.sha256(snap.read_bytes()).hexdigest(),'archives':checks}))
'''.replace('ROW',repr(r)).replace('REUSE_BACKUP',repr(a.problem=='ac1'))
d=json.loads(ssh(r['external_ip'],code,timeout=660));bucket=w.ops.storage.bucket('sk7524-tinker-tpu-us-central1');prefix='ray-training/'+r['run_id']+'/'
for name,meta in d['archives'].items():
 b=bucket.get_blob(prefix+name);assert b and b.size==meta['size'] and b.md5_hash==meta['md5'],name
b=bucket.get_blob(prefix+'client/tinker_log/'+r['run_id']+f"/puct_sampler_step_{d['checkpoint']['batch']:06d}.json");import hashlib
assert hashlib.sha256(b.download_as_bytes()).hexdigest()==d['snapshot_sha256']
d['verified_time']=time.time();save(OUT/(a.problem+'-twohost-handoff-ready.json'),d);print(json.dumps(d))
