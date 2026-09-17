import concurrent.futures,subprocess,os,json
from pathlib import Path
p=Path('.science/placement-assessment-20260917/qwen-gemma-parity');cpus=sorted(os.sched_getaffinity(0))
def worker(i):
 for model in ['qwen','gemma']:
  case=['ibm01','ibm04','ibm08','ibm18'][i]
  r=subprocess.run(['taskset','-c',','.join(map(str,cpus[i*4:i*4+4])),'python3',str(p/'cpu_case.py'),model,case],env=dict(os.environ,PYTHONPATH=str(Path.cwd())),capture_output=True,text=True,timeout=285)
  print(r.stdout if r.returncode==0 else r.stderr[-1600:],flush=True)
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as e:list(e.map(worker,range(4)))
