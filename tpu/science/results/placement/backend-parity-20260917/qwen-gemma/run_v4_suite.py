import concurrent.futures,json,subprocess,time
from pathlib import Path
folder=Path('.science/placement-assessment-20260917/qwen-gemma-parity');stamp=str(int(time.time()));base='tpuswarm-v4-64-central2-qwen35-erdos-406';cfg='/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh/'+base
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','LogLevel=ERROR','-F',cfg,base];remote='/tmp/placement-qwen-gemma-parity-'+stamp+'.py';subprocess.run(cmd+['cat > '+remote],input=(folder/'remote_probe.py').read_text(),text=True,check=True,timeout=20)
def worker(chip):
 case=['ibm01','ibm04','ibm08','ibm18'][chip]
 for model,variant in [('qwen','original'),('gemma','original'),('qwen','barrier'),('gemma','barrier')]:
  name=model+'-'+variant+'-'+case;source=folder.parent/(model+'-best.py') if variant=='original' else folder/(model+'-barrier.py')
  spec=dict(chip=chip,case=case,name=name+'-'+stamp,source=source.read_text())
  r=subprocess.run(cmd+['/home/gcpuser/.cache/skyrl-ray-code/7ab6a72ce38b94369298f3c0e940722cbc2211cb3b15a7c55e14029d45278fbb/.science/venv/bin/python',remote],input=json.dumps(spec),capture_output=True,text=True,timeout=325)
  (folder/(name+'-remote.json')).write_text(r.stdout);(folder/(name+'-remote.stderr')).write_text(r.stderr)
  if r.returncode:print(name,'ERROR',r.stderr[-1600:],flush=True);continue
  x=json.loads(r.stdout);y=x.get('result')
  print(json.dumps(dict(name=name,code=x['returncode'],valid=y['correctness'] if y else None,metrics={k:y['metrics'].get(k) for k in ['case','proxy_cost','candidate_wall_seconds','grading_seconds']} if y else None,error=y['msg'] if y else x['log'][-1000:])),flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as p:list(p.map(worker,range(4)))
