"""Offline observation-only migration, with original grade records retained."""
import hashlib,json,os,sys,time
from pathlib import Path
from tpu.science.feedback import observation,diagnostic_message
from tpu.science.bootstrap import save
from tpu.science import feedback
run=sys.argv[1]
p=Path(sys.argv[2]).resolve() if len(sys.argv)>2 else Path.home()/'.cache'/run/'runs'/run/'client/bootstrap'
assert run.startswith('science-placement-v4-')
assert not (p/'layer-1-plan.json').exists() and not (p/'complete.json').exists()
assert 0 < len(list(p.glob('layer-0/group-*/generation.json'))) <= 16
# Require the producer to have stopped, rather than racing writes in an active journal.
for proc in Path('/proc').iterdir():
 if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
 try:args=proc.joinpath('cmdline').read_bytes().decode(errors='replace').split('\0')
 except (FileNotFoundError,PermissionError):continue
 if 'tpu.science.bootstrap' in args and any(run in a for a in args):raise RuntimeError('bootstrap is still running')
audit=p/'feedback-migrations'/str(time.time_ns());audit.mkdir(parents=True)
changed=[]
for f in sorted(p.glob('layer-0/group-*/grade-*.json')):
 old=json.loads(f.read_text())
 if 'cases' not in old['metrics']:continue
 result=dict(reward=old['reward'],correctness=old['correctness'],msg=old['message'],metrics=old['metrics'])
 row=dict(old,feedback=observation('placement',result),message=diagnostic_message('placement',result))
 if row==old:continue
 name=f.relative_to(p)
 save(audit/name,old)
 assert {k:v for k,v in row.items() if k not in ('message','feedback')}=={k:v for k,v in old.items() if k not in ('message','feedback')}
 save(f,row)
 changed.append({'path':str(name),'before':hashlib.sha256(json.dumps(old,sort_keys=True).encode()).hexdigest(),'after':hashlib.sha256(json.dumps(row,sort_keys=True).encode()).hexdigest()})
save(audit/'manifest.json',{'operation':'diagnostic feedback only; unchanged generations, code, rewards and metrics','formatter_sha256':hashlib.sha256(Path(feedback.__file__).read_bytes()).hexdigest(),'contract_sha256':json.loads((p/'contract.json').read_text())['sha256'],'changed':changed})
print(json.dumps({'run':run,'changed':len(changed),'audit':str(audit)}))
