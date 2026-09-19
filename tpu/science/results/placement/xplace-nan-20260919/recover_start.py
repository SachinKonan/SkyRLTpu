from pathlib import Path
import sys,time,json,hashlib,subprocess
import numpy as np,torch
root=Path.cwd();sys.path.insert(0,str(root));sys.path.insert(0,str(root/'.science/challenge-probe'))
from macro_place.loader import load_benchmark_from_dir
from tpu.science.challenge_contract import problem_from_native
from tpu.science.challenge_seed_jax import legalize
from tpu.science.full17_eval import cpu_case,save
case=sys.argv[1];out=root/'tpu/science/results/placement/full17-20260919';audit=root/'.science/xplace-nan-audit-20260919'/case;audit.mkdir(exist_ok=True)
torch.set_num_threads(4)
b,plc=load_benchmark_from_dir(str(root/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/case))
p=problem_from_native(b,plc)
if case == 'ibm14':
 import importlib.util
 from tpu.science.placement_gpu_suite import require_finite_xplace_log
 recovery=root/'.science/xplace-nan-audit-20260919/guarded'
 require_finite_xplace_log((recovery/'run.log').read_text())
 spec=importlib.util.spec_from_file_location('bridge',root/'.science/archgen-cuda-run/submissions/bookshelf_to_lefdef.py')
 module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
 raw=module._parse_def_or_pl(recovery/'design_design_gp.def',b)
 np.save(audit/'raw_positions.npy',raw,allow_pickle=False)
else:
 raw=np.load(out/f'xplace-{case}/raw_positions.npy',allow_pickle=False)
assert np.isfinite(raw).all()
print('raw',case,raw.shape,'minmax',raw.min(),raw.max(),'unchanged',np.sum(np.all(raw==p['initial_positions'],axis=1)),flush=True)
p['initial_positions']=raw.copy();p['initial_positions'][p['fixed']]=b.macro_positions.numpy()[p['fixed']]
start=time.monotonic();positions=legalize(p,42,time_budget_s=7200)['positions'];elapsed=time.monotonic()-start
np.save(audit/'positions.npy',positions,allow_pickle=False)
subprocess.run([str(root/'.science/venv-cuda/bin/python'),'-m','tpu.science.challenge_score_child','--root',str(root),'--case',case,'--positions',str(audit/'positions.npy'),'--result',str(audit/'score.json')],check=True,timeout=180)
p['initial_positions']=positions
inputs=out/'inputs';inputs.mkdir(exist_ok=True);np.savez_compressed(inputs/f'{case}.npz',**p);np.save(inputs/f'{case}-start.npy',positions,allow_pickle=False)
old=json.loads((out/f'xplace-{case}/report.json').read_text())
meta={'case':case,'provenance':'same original finite Xplace raw output; deterministic legalizer with 7200s preparation allowance','xplace_report':str(out/f'xplace-{case}/report.json'),'recovery_score':str(audit/'score.json'),'recovery_legalizer_seconds':elapsed,'xplace_seconds':old['candidate_wall_seconds']+elapsed,'problem_sha256':hashlib.sha256((inputs/f'{case}.npz').read_bytes()).hexdigest()}
if case == 'ibm14':
 meta['provenance']='Xplace with positive finite Nesterov step guard; original hyperparameters and seed 42; 7200s deterministic legalization'
 meta['optimizer_patch']='tpu/science/results/placement/xplace-nan-20260919/nesterov-positive-step.patch'
 meta['guarded_xplace_log']=str(root/'.science/xplace-nan-audit-20260919/guarded/run.log')
 meta['xplace_seconds'] += 17.275
save(inputs/f'{case}.json',meta);save(inputs/f'{case}-score.json',json.loads((audit/'score.json').read_text()));save(audit/'recovery.json',meta)
print('VALID_START',case,elapsed,flush=True)
for model in ['gemma','qwen','muse']:
 if not (out/f'{model}-{case}/report.json').exists():cpu_case(out,case,model)
print('MODEL_REPLAYS_COMPLETE',case,flush=True)
