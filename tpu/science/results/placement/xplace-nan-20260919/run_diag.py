from pathlib import Path
import os,sys,subprocess,json
sys.path.insert(0,str(Path.cwd()))
from tpu.science.isolation import command,python_mounts
root=Path.cwd(); audit=root/'.science/xplace-nan-audit-20260919'; xp=audit/'xplace'; py=root/'.science/venv-cuda/bin/python'
bench=root/'tpu/science/results/placement/full17-20260919/xplace-ibm14/xplace/bench'
mode=sys.argv[1] if len(sys.argv)>1 else 'instrumented'
out=audit/mode;out.mkdir(exist_ok=True)
args=[str(py),'main.py','--custom_path=benchmark:custom_lefdef,lef:/input/design.lef,def:/input/design.def,design_name:design','--load_from_raw=True','--global_placement=True','--legalization=False','--detail_placement=False','--use_filler=False','--target_density=0.8','--inner_iter=1200','--use_route_force=True','--use_cell_inflate=False','--route_weight=0.01','--congest_weight=0.01','--num_route_iter=20','--num_bin_x=128','--num_bin_y=128','--write_placement=True','--write_global_placement=True','--output_dir=/output','--output_prefix=design','--draw_placement=False','--num_threads=4','--final_route_eval=False','--mixed_size=True','--seed=42']
if mode=='no-route':args[args.index('--use_route_force=True')]='--use_route_force=False'
env={'CUDA_VISIBLE_DEVICES':'GPU-2034d210-fcce-31f7-2173-aebf35a3995a','PATH':str(py.parent)+':/usr/local/cuda-12.6/bin:/usr/bin:/bin','CUDA_HOME':'/usr/local/cuda-12.6','CUBLAS_WORKSPACE_CONFIG':':4096:8','MPLCONFIGDIR':'/tmp/matplotlib','XDG_CACHE_HOME':'/tmp/cache','NUMBA_CACHE_DIR':'/tmp/numba','PYTHONUNBUFFERED':'1'}
cmd=command(args,readonly=python_mounts(py)+[(root/'.science/xplace-cuda',root/'.science/xplace-cuda'),(bench,'/input'),('/usr/local','/usr/local'),('/sys','/sys')],writable=[(xp,xp),(out,'/output')],env=env,cwd=str(xp))
i=cmd.index('--dev')+2
cmd[i:i]=['--dev-bind','/dev/nvidia1','/dev/nvidia1','--dev-bind','/dev/nvidiactl','/dev/nvidiactl','--dev-bind','/dev/nvidia-uvm','/dev/nvidia-uvm','--tmpfs','/dev/shm']
with (out/'run.log').open('wb') as f:
 r=subprocess.run(['taskset','-c','64-67',*cmd],stdout=f,stderr=subprocess.STDOUT,timeout=300)
(out/'exit.json').write_text(json.dumps({'returncode':r.returncode,'args':args}))
print(mode,r.returncode)
