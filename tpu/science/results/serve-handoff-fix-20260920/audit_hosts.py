"""Read-only post-stop checks; never kill a different run's processes."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[4];HERE=Path(__file__).resolve().parent
rows=json.loads((HERE/'before-stop.json').read_text())['jobs']
clusters={r['old_worker'] for r in rows if r['old_worker'] and r['old_status'] in ('RUNNING','STARTING')}
clusters.add('tpuswarm-v4-64-central2-qwen35-erdos-678')
script=(ROOT/'tpu/results/native-training-recovery-20260913/clean_host_audit.py').read_text()
def check(pair):
 cluster,rank=pair;host=cluster+(f'-worker{rank}' if rank else '')
 cmd=['ssh','-F','/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh/'+cluster,'-o','BatchMode=yes','-o','ConnectTimeout=8',host,'python3 -']
 try:
  p=subprocess.run(cmd,input=script,text=True,capture_output=True,timeout=25)
  return dict(cluster=cluster,rank=rank,exit=p.returncode,stdout=p.stdout,stderr=p.stderr)
 except subprocess.TimeoutExpired:return dict(cluster=cluster,rank=rank,exit=124,stderr='SSH timeout')
with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:results=list(pool.map(check,[(c,r) for c in sorted(clusters) for r in range(8)]))
(HERE/'clean-hosts.json').write_text(json.dumps(results,indent=2)+'\n')
for c in sorted(clusters):
 print(c,[(r['rank'],r['exit']) for r in results if r['cluster']==c],flush=True)
