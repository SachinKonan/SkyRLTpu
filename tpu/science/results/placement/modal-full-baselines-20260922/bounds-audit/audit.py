import sys,json,contextlib,io
from pathlib import Path
import numpy as np
import torch
torch.set_num_threads(4);torch.set_num_interop_threads(1)
root=Path.cwd();sys.path.insert(0,str(root/'.science/challenge-probe'))
from macro_place.loader import load_benchmark_from_dir
from macro_place.utils import validate_placement
from macro_place.objective import compute_proxy_cost
out=root/'tpu/science/results/placement/modal-full-baselines-20260922/bounds-audit';rows=[]
for p in sorted((root/'.science/modal-baselines/results').glob('archgen-*/positions.npy')):
 case=p.parent.name.split('-')[1]
 with contextlib.redirect_stdout(io.StringIO()):b,plc=load_benchmark_from_dir(str(root/'.science/challenge-probe/external/MacroPlacement/Testcases/ICCAD04'/case))
 pos=torch.from_numpy(np.load(p));sizes=b.macro_sizes;canvas=torch.tensor([b.canvas_width,b.canvas_height],dtype=pos.dtype)
 excess=torch.maximum(-(pos-sizes/2),pos+sizes/2-canvas);bad=(excess>0).any(dim=1)
 valid,errors=validate_placement(pos,b);fixed=b.macro_fixed
 row=dict(case=case,valid=valid,errors=errors,canvas=canvas.tolist(),bad_count=int(bad.sum()),max_excess=float(excess.clamp_min(0).max()),max_excess_canvas_ulps=float((excess.clamp_min(0)/(torch.nextafter(canvas,torch.full_like(canvas,float('inf')))-canvas)).max()),fixed_unchanged=bool(torch.equal(pos[fixed],b.macro_positions[fixed])),bad_indices=torch.where(bad)[0].tolist())
 # Diagnostic score of original positions: no coordinate modification, NOT official validity.
 with contextlib.redirect_stdout(io.StringIO()):metrics=compute_proxy_cost(pos,b,plc)
 row['diagnostic_metrics']={k:float(v) for k,v in metrics.items()};rows.append(row);(out/(case+'.json')).write_text(json.dumps(row,indent=2));print(json.dumps(row),flush=True)
(out/'summary.json').write_text(json.dumps(rows,indent=2))
