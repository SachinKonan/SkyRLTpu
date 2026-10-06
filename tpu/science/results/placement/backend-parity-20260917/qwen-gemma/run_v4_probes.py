import concurrent.futures,json,subprocess,time
from pathlib import Path
folder=Path('.science/placement-assessment-20260917/qwen-gemma-parity');stamp=str(int(time.time()));base='tpuswarm-v4-64-central2-qwen35-erdos-406';cfg='/scratch/gpfs/ZHUANGL/sk7524/tpuswarm-state/sky-home-v6e32/.sky/generated/ssh/'+base
cmd=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=8','-o','LogLevel=ERROR','-F',cfg,base];remote='/tmp/placement-qwen-gemma-parity-'+stamp+'.py';subprocess.run(cmd+['cat > '+remote],input=(folder/'remote_probe.py').read_text(),text=True,check=True,timeout=20)

def run_one(item):
 chip,name=item
 spec=dict(chip=chip,case='ibm01',name=name+'-'+stamp,source=(folder/(name+'.py')).read_text())
 r=subprocess.run(cmd+['/home/gcpuser/.cache/skyrl-ray-code/7ab6a72ce38b94369298f3c0e940722cbc2211cb3b15a7c55e14029d45278fbb/.science/venv/bin/python',remote],input=json.dumps(spec),capture_output=True,text=True,timeout=325)
 (folder/(name+'-remote.json')).write_text(r.stdout);(folder/(name+'-remote.stderr')).write_text(r.stderr)
 r.check_returncode();x=json.loads(r.stdout)
 logs=subprocess.run(cmd+['cat '+x['folder']+'/evaluation/candidate.log'],capture_output=True,text=True,timeout=20)
 (folder/(name+'-tpu.log')).write_text(logs.stdout)
 print(name,x['returncode'],logs.stdout,flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
 list(pool.map(run_one,enumerate(['qwen-gradient','qwen-gradient-barrier','gemma-gradient','gemma-gradient-barrier'])))
 list(pool.map(run_one,enumerate(['qwen-gradient-split','qwen-trace','qwen-trace-barrier'])))
