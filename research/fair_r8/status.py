"""Read only this experiment's lab units and artifacts; write a dated snapshot."""
import argparse
import json
from pathlib import Path
import shlex
import subprocess

SSH=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-F',
     '/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh/tpuswarm-v5p32-east5a-erdos-3242',
     'tpuswarm-v5p32-east5a-erdos-3242']

CODE='''import datetime,json,pathlib,subprocess,os
root=pathlib.Path('/dev/shm/fair-r8-20261009');runs={}
assert os.uname().nodename=='t1v-n-51b3bdd2-w-0'
for w,s in [(16,0),(16,1),(32,0),(32,1)]:
 name=f'field-w{w}-s{s}';d={}
 for f in ['best.json','complete.json']:
  p=root/name/f
  if p.exists():d[f]=json.loads(p.read_text())
 p=root/name/'history.jsonl'
 if p.exists():d['latest_validation']=json.loads(p.read_text().splitlines()[-1])
 d['train_unit']=subprocess.run(['systemctl','show',f'fair-r8-train-w{w}-s{s}','-p','ActiveState','-p','Result','-p','MemoryCurrent'],capture_output=True,text=True).stdout
 p=root/f'numerical-w{w}-s{s}'/'report.json'
 if p.exists():d['numerical']=json.loads(p.read_text())
 runs[name]=d
post=subprocess.run(['systemctl','show','fair-r8-post-training','-p','ActiveState','-p','Result'],capture_output=True,text=True).stdout
extra={}
for name in ['normal-audit-001/report.json','normal-study-001.receipt.json','frozen-checkpoints.json']:
 p=root/name
 if p.exists():extra[name]=json.loads(p.read_text())
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),host=os.uname().nodename,runs=runs,post_unit=post,extra=extra)))
'''


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);a=p.parse_args()
    result=subprocess.run(SSH+['python3 -c '+shlex.quote(CODE)],check=True,capture_output=True,text=True)
    data=json.loads(result.stdout);Path(a.out).write_text(json.dumps(data,indent=2))
    print(json.dumps(dict(utc=data['utc'],post=data['post_unit'],
        models={k:dict(latest=v.get('latest_validation'),complete='complete.json' in v,
                       numerical_pass=v.get('numerical',{}).get('proceed64')) for k,v in data['runs'].items()},
        normal_quality='normal-audit-001/report.json' in data['extra'])))
