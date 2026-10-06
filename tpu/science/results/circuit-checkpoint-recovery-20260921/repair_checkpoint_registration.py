import json,hashlib,base64,sqlite3,subprocess,sys,time,os
from pathlib import Path
root=Path.home()/'.cache/fresh-v4-gemma-circuit-grpo-lr4e5-s1-20260919';r=root/'runs'/root.name
log=r/'client/tinker_log'/root.name
proof=json.load(open(root/'checkpoint-validation-20260921.json'))
assert proof['training']['lora_sha256']==proof['sampler_weights']['lora_sha256']
assert proof['training']['optimizer_scalars']["['step'].value"]==1
assert proof['training']['optimizer_scalars']["['opt_state']['inner_state'][0]['count'].value"]==1
assert json.load(open(log/'global_step.json'))=={'global_step':1}
assert hashlib.sha256((log/'puct_sampler_step_000001.json').read_bytes()).hexdigest()==proof['puct']['puct_sampler_step_000001.json']['sha256']
state=subprocess.check_output(['systemctl','show','skyrl-circuit10-gemma-a5','-p','ActiveState','--value'],text=True).strip()
assert state in ('failed','inactive'),state
prefix='gs://sk7524-tinker-tpu-us-central2/ray-training/'+root.name
receipt={'time':time.time(),'run_id':root.name,'step':1,'verified_archives':[],'proof':{'lora_sha256':proof['training']['lora_sha256'],'optimizer_step':1,'puct_sha256':proof['puct']['puct_sampler_step_000001.json']['sha256']}}
for family in ['', 'sampler_weights']:
 p=r/'checkpoints/model_a0eb148b'/family/'000001.tar.gz';uri=prefix+'/checkpoints/model_a0eb148b/'+(family+'/' if family else '')+p.name
 meta=json.loads(subprocess.check_output(['gcloud','storage','objects','describe',uri,'--format=json'],text=True,stderr=subprocess.DEVNULL,timeout=60))
 h=hashlib.md5()
 with p.open('rb') as f:
  while chunk:=f.read(8*1024*1024):h.update(chunk)
 digest=base64.b64encode(h.digest()).decode()
 assert int(meta['size'])==p.stat().st_size and meta['md5_hash']==digest,(family,meta)
 receipt['verified_archives'].append({'uri':uri,'generation':meta['generation'],'size':int(meta['size']),'md5_hash':digest})
db=r/'tinker.db';c=sqlite3.connect(db)
assert c.execute('PRAGMA quick_check').fetchone()[0]=='ok'
assert c.execute("SELECT request_id FROM futures WHERE model_id='model_a0eb148b' AND request_type='OPTIM_STEP' AND status='COMPLETED'").fetchall()==[(125,)]
assert c.execute("SELECT count(*) FROM futures WHERE model_id='model_a0eb148b' AND request_type='FORWARD_BACKWARD' AND request_id>125").fetchone()[0]==0
assert c.execute("SELECT status FROM checkpoints WHERE model_id='model_a0eb148b' AND checkpoint_id='000001' AND checkpoint_type='SAMPLER'").fetchone()==('COMPLETED',)
index=log/'member_gemma/checkpoints.jsonl';assert not index.exists()
backup=root/'pre-repair-step1.db';assert not backup.exists()
bc=sqlite3.connect(backup);c.backup(bc);bc.close()
old=c.execute("SELECT status,error_message FROM checkpoints WHERE model_id='model_a0eb148b' AND checkpoint_id='000001' AND checkpoint_type='TRAINING'").fetchone()
assert old[0]=='FAILED' and 'checkpoint mirror upload failed' in old[1],old
receipt['previous_checkpoint_registration']={'status':old[0],'error':old[1]}
updated=c.execute("UPDATE checkpoints SET status='COMPLETED', completed_at=CURRENT_TIMESTAMP, error_message=NULL WHERE model_id='model_a0eb148b' AND checkpoint_id='000001' AND checkpoint_type='TRAINING' AND status='FAILED'")
assert updated.rowcount==1;c.commit();c.close()
row={'name':'000001','batch':1,'state_path':'tinker://model_a0eb148b/weights/000001','sampler_path':'tinker://model_a0eb148b/sampler_weights/000001'}
with index.open('x') as f:f.write(json.dumps(row)+'\n')
# Preserve the failed request as historical evidence. Only materialization status
# changes after both archive checksums and the optimizer step are verified.
sys.path.insert(0,'/home/gcpuser/.cache/skyrl-ray-code/06dd9c468c31327230653a8983121d3ce5748aae422cdc38fb1366621ce0542d')
from tpu.swarm.ray_train.database_snapshot import create_snapshot
create_snapshot(db,r/'tinker-backup.db')
receipt['checkpoint_index']=row;receipt['state']='local_repaired';(root/'checkpoint-registration-repair.json').write_text(json.dumps(receipt,indent=2))
env=dict(os.environ,CLOUDSDK_STORAGE_PROCESS_COUNT='1',CLOUDSDK_STORAGE_THREAD_COUNT='1')
for local,dest in [(r/'tinker-backup.db',prefix+'/tinker-backup.db'),(r/'client',prefix+'/')]:
 cmd=['gcloud','storage','cp']+(['--recursive'] if local.is_dir() else [])+[str(local),dest]
 q=subprocess.run(cmd,env=env,capture_output=True,text=True,timeout=600)
 assert q.returncode==0,(q.returncode,q.stderr[-1000:])
receipt['state']='published';receipt['finished']=time.time();(root/'checkpoint-registration-repair.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
