"""Derive a fixed nearest-cluster score; fit nothing to public test data."""
import argparse,hashlib,json,shutil
from pathlib import Path
import numpy as np
from score import nll

p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);a=p.parse_args()
source=Path(a.source);out=Path(a.out);out.mkdir(exist_ok=False);dest=out/'density';dest.mkdir()
for name in ['config.json','normalization.json','best.msgpack','best.json','source.py','splits.npz','linear_probe.json']:
    if (source/name).exists():shutil.copy2(source/name,out/name)
cfg=json.loads((out/'config.json').read_text());cfg['score_variant']='nearest_standardized_distance';(out/'config.json').write_text(json.dumps(cfg,indent=2))
with np.load(source/'density/model.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
d['score_mode']=np.array('nearest_standardized_distance');np.savez(dest/'model.npz',**d)
report=json.loads((source/'density/report.json').read_text())
report.pop('normal_audit_by_cosmology',None);report.pop('normal_audit_thresholds',None)
for split in ['train','val','calibration','audit']:
    shutil.copy2(source/f'density/{split}_latent.npy',dest/f'{split}_latent.npy')
    if split in ['calibration','audit']:
        scores=nll(np.load(dest/f'{split}_latent.npy'),d);np.save(dest/f'{split}_scores.npy',scores)
        report[split]={'count':len(scores),'mean':float(scores.mean()),'std':float(scores.std()),'quantiles':np.quantile(scores,[.5,.9,.95,.99,.999]).tolist()}
cal=np.load(dest/'calibration_scores.npy');audit=np.load(dest/'audit_scores.npy')
thresholds=np.quantile(cal,[.95,.99,.999]);report['normal_audit_thresholds']=thresholds.tolist()
report['normal_audit_exceedance_counts']=[int((audit>t).sum()) for t in thresholds]
report['method']='Minimum squared standardized distance to existing normal-fitted clusters; same K/centers/variances as parent; no mixing weights or determinant term'
report['parent_density_sha256']=hashlib.sha256((source/'density/model.npz').read_bytes()).hexdigest()
code=Path(__file__).parent
report['runtime_files']={name:hashlib.sha256((code/name).read_bytes()).hexdigest() for name in ['train.py','features.py','summary_train.py']}
report['frozen_files']={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['config.json','normalization.json','best.msgpack','source.py','density/model.npz']}
report['score_source_sha256']=hashlib.sha256((code/'score.py').read_bytes()).hexdigest()
(dest/'report.json').write_text(json.dumps(report,indent=2))
print(json.dumps({'output':str(out),'audit_exceedances':report['normal_audit_exceedance_counts']}),flush=True)
