import json,time,subprocess,os
from pathlib import Path
out=Path('tpu/science/results/placement/gpu-suite-20260916')
state=out/'a40-remaining-status.json'
state.write_text(json.dumps({'phase':'waiting_for_ibm01_baselines','remaining':6}))
while not all((out/(m+'-ibm01')/'report.json').exists() for m in ['abuplace','archgen']):
 pid=json.loads((out/'human-a40-launch.json').read_text())['pid']
 try:os.kill(pid,0)
 except ProcessLookupError:raise RuntimeError('Initial A40 supervisor stopped before producing both reports')
 time.sleep(15)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PLACEMENT_GPU_NODE='/dev/nvidia0',OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for case in ['ibm04','ibm08','ibm18']:
 for method,repo in [('abuplace','.science/abuplace'),('archgen','.science/archgen-cuda-run')]:
  state.write_text(json.dumps({'phase':'running','method':method,'case':case,'started_unix':time.time()}))
  if (out/(method+'-'+case)/'report.json').exists():continue
  subprocess.run(['.science/venv-cuda/bin/python','tpu/science/placement_gpu_suite.py','--method',method,'--case',case,'--repository',str(Path(repo).resolve()),'--xplace-root',str(Path('.science/xplace-cuda').resolve()),'--output',str(out/(method+'-'+case)),'--seconds','3450'],env=env,check=False)
state.write_text(json.dumps({'phase':'complete','finished_unix':time.time()}))
