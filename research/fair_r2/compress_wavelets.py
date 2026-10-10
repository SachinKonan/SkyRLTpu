"""Train-only wavelet PCA, retaining the original 72 coordinates unchanged."""
import argparse
from pathlib import Path
import numpy as np
from common import load_features, write_json, sha

p=argparse.ArgumentParser();p.add_argument('--features',required=True);p.add_argument('--out',required=True);p.add_argument('--dim',type=int,default=16);a=p.parse_args()
root=Path(a.features);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
features=load_features(root);x=features['train'][:,72:]
offset=np.median(x,0);scale=np.maximum(np.subtract(*np.quantile(x,[.75,.25],axis=0))/1.349,1e-6)
x=np.arcsinh((x-offset)/scale);mean=x.mean(0);std=np.maximum(x.std(0),1e-6);x=(x-mean)/std
e,u=np.linalg.eigh(np.cov(x,rowvar=False));order=np.argsort(-e);e=e[order];basis=u[:,order[:a.dim]]
def transform(v):
    z=(np.arcsinh((v[:,72:]-offset)/scale)-mean)/std
    return np.column_stack([v[:,:72],z@basis]).astype(np.float32)
np.savez(out/'pca.npz',offset=offset,scale=scale,mean=mean,std=std,basis=basis)
for name,x in features.items():np.save(out/(name+'_latent.npy'),transform(x))
(out/'probes').mkdir()
for path in (root/'probes').glob('*.npz'):
    with np.load(path) as f:np.savez(out/'probes'/path.name,**{k:f[k] if k=='ids' else transform(f[k]) for k in f.files})
write_json(out/'manifest.json',dict(kind='TRAIN_ONLY_PCA',source=str(root),dimensions=72+a.dim,wavelet_pca_dimensions=a.dim,variance_fraction=float(e[:a.dim].sum()/e.sum()),pca_sha256=sha(out/'pca.npz'),feature_hashes={p.name:sha(p) for p in out.glob('*.npy')}))
print((out/'manifest.json').read_text())
