"""Evaluate frozen R1/R2 fits on new morphology families, never public test."""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import jax
from common import low_fpr_metric,write_json,sha
from flow import predictor as flow_predictor
from class_gaussian import predictor as gaussian_predictor

p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--r1',required=True);p.add_argument('--out',required=True);a=p.parse_args()
root=Path(a.root);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
with np.load(root/'morphology-probes/features216/morphology_r2.npz') as f:
    ids=f['ids'];features={k:f[k] for k in f.files if k!='ids'}
with np.load(root/'features88/pca.npz') as f:pc={k:f[k] for k in f.files}
def subset(x,dim):
    if dim==72:return x[:,:72]
    if dim==216:return x
    if dim==88:
        z=(np.arcsinh((x[:,72:]-pc['offset'])/pc['scale'])-pc['mean'])/pc['std']
        return np.column_stack([x[:,:72],z@pc['basis']])
    raise ValueError(dim)
models=[]
for path in sorted(root.glob('flow*')):
    if not (path/'report.json').exists():continue
    cfg=json.loads((path/'config.json').read_text());dim=cfg['dim']+(20 if cfg['condition']=='power' else 0)
    models.append((path.name,path,dim,'flow'))
for name,dim in [('gaussian72',72),('gaussian216',216)]:models.append((name,root/name,dim,'gaussian'))
all_scores={};calibrations={};report={};start=time.monotonic()
for name,path,dim,kind in models:
    fn=flow_predictor(path) if kind=='flow' else gaussian_predictor(path)
    with np.load(path/'calibration_scores.npz') as f:cal={k:f[k] for k in f.files}
    metrics={};results={}
    for family,x in features.items():
        s=fn(subset(x,dim));metrics[family]={k:low_fpr_metric(cal[k],v) for k,v in s.items()}
        results.update({family+'__'+k:v for k,v in s.items()})
    np.savez(out/(name+'.npz'),**results);report[name]=metrics
    for key,c in cal.items():
        all_scores[name+'/'+key]={family:results[family+'__'+key] for family in features}
        calibrations[name+'/'+key]=c
    print(name,json.dumps(metrics),flush=True);del fn;jax.clear_caches()
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
from score import nll
for name in ('physics','physics-distance'):
    path=Path(a.r1)/name/'density'
    with np.load(path/'model.npz') as f:model={k:f[k] for k in f.files}
    cal=np.load(path/'calibration_scores.npy');result={family:nll(x[:,:72],model) for family,x in features.items()}
    np.savez(out/('r1-'+name+'.npz'),**result);report['r1-'+name]={family:low_fpr_metric(cal,v) for family,v in result.items()}
for condition in ('none','cosmology','power'):
    score='minimum_base_energy' if condition=='cosmology' else 'base_energy'
    names=[f'flow-{condition}-asinh-s{seed}/{score}' for seed in range(3)]
    cal=np.mean([calibrations[n] for n in names],0)
    result={family:np.mean([all_scores[n][family] for n in names],0) for family in features}
    np.savez(out/('ensemble-'+condition+'.npz'),**result)
    report['ensemble-'+condition]={family:low_fpr_metric(cal,v) for family,v in result.items()}
write_json(out/'report.json',dict(kind='NEW_ENGINEERED_PROBES_ON_REUSED_AUDIT_NOT_OFFICIAL_OOD',probe_manifest_sha256=sha(root/'morphology-probes/manifest.json'),count=len(ids),seconds=time.monotonic()-start,results=report));print('COMPLETE',time.monotonic()-start,flush=True)
