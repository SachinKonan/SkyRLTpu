import importlib.util,json,torch
from pathlib import Path
root=Path.cwd()
def run(path,name):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 x=torch.tensor([[1000000.,1000000.]],requires_grad=True);x.grad=torch.zeros_like(x)
 opt=m.NesterovOptimizer([x],lr=1e-6)
 def closure(y):return (y*1e-4).sum(),torch.full_like(y,1e-4)
 for _ in range(3):opt.step(closure)
 return bool(torch.isfinite(x).all())
old=run(root/'.science/xplace-cuda/src/nesterov_optimizer.py','old_optimizer')
new=run(root/'.science/xplace-nan-audit-20260919/xplace/src/nesterov_optimizer.py','fixed_optimizer')
assert not old and new,(old,new)
print(json.dumps({'original_finite':old,'guarded_finite':new}))
