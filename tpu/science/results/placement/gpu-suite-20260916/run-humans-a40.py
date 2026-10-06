import json,time,subprocess,os
from pathlib import Path
out=Path('tpu/science/results/placement/gpu-suite-20260916')
while not all((out/('xplace-'+c)/'report.json').exists() for c in ['ibm01','ibm04','ibm08','ibm18']):
 pid=json.loads((out/'xplace-launch.json').read_text())['pid']
 try:os.kill(pid,0)
 except ProcessLookupError:raise RuntimeError('Xplace driver exited before all four reports')
 time.sleep(15)
env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',PLACEMENT_GPU_NODE='/dev/nvidia0',OMP_NUM_THREADS='16',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
for method,repo in [('abuplace','.science/abuplace'),('archgen','.science/archgen-cuda-run')]:
 subprocess.run(['.science/venv-cuda/bin/python','tpu/science/placement_gpu_suite.py','--method',method,'--case','ibm01','--repository',str(Path(repo).resolve()),'--xplace-root',str(Path('.science/xplace-cuda').resolve()),'--output',str(out/(method+'-ibm01')),'--seconds','3450'],env=env,check=False)
