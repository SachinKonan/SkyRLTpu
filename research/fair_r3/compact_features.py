"""Train-only PCA of spatial descriptors, preserving 72 baseline coordinates."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_r2'))
from common import load_features,write_json,sha


def fit_part(train,indices,dim):
    x=train[:,indices]
    offset=np.median(x,0)
    scale=np.maximum(np.subtract(*np.quantile(x,[.75,.25],axis=0))/1.349,1e-6)
    z=np.arcsinh((x-offset)/scale);mean=z.mean(0);std=np.maximum(z.std(0),1e-6)
    z=(z-mean)/std;e,u=np.linalg.eigh(np.cov(z,rowvar=False));order=np.argsort(-e)
    e=e[order];basis=u[:,order[:dim]]
    sign=np.sign(basis[np.argmax(abs(basis),axis=0),np.arange(dim)])
    basis=basis*np.where(sign==0,1,sign)
    return dict(indices=indices,offset=offset,scale=scale,mean=mean,std=std,basis=basis),float(e[:dim].sum()/e.sum())


def transform(x,parts):
    values=[x[:,:72]]
    for m in parts:
        z=(np.arcsinh((x[:,m['indices']]-m['offset'])/m['scale'])-m['mean'])/m['std']
        values.append(z@m['basis'])
    return np.column_stack(values).astype(np.float32)


def load_parts(path):
    with np.load(path,allow_pickle=False) as f:
        count=int(f['count'])
        return [{k.removeprefix(f'part{i}_'):f[k] for k in f.files if k.startswith(f'part{i}_')}
                for i in range(count)]


def main(a):
    root=Path(a.features);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    features=load_features(root)
    if features['train'].shape[1]!=165:raise ValueError('Expected 72+45+48 features')
    plans={'morph':[(np.arange(72,117),16)],'regional':[(np.arange(117,165),16)],
           'joint':[(np.arange(72,117),8),(np.arange(117,165),8)]}
    fitted=[fit_part(features['train'],ix,dim) for ix,dim in plans[a.kind]]
    parts=[p for p,_ in fitted]
    np.savez(out/'pca.npz',count=len(parts),**{f'part{i}_{k}':v for i,p in enumerate(parts) for k,v in p.items()})
    for name,x in features.items():np.save(out/f'{name}_latent.npy',transform(x,parts))
    probes=out/'probes';probes.mkdir()
    for path in (root/'probes').glob('*.npz'):
        with np.load(path) as f:
            np.savez(probes/path.name,**{k:f[k] if k=='ids' else transform(f[k].astype(np.float64),parts) for k in f.files})
    # Serialization must preserve the fitted transform.
    assert np.array_equal(transform(features['val'][:17],parts),transform(features['val'][:17],load_parts(out/'pca.npz')))
    write_json(out/'manifest.json',dict(kind='TRAIN_ONLY_SPATIAL_PCA',variant=a.kind,dimensions=88,
        retained_variance_fractions=[fraction for _,fraction in fitted],source=str(root),
        pca_sha256=sha(out/'pca.npz'),source_sha256=sha(__file__),
        feature_hashes={p.name:sha(p) for p in out.glob('*.npy')},
        public_test_read=False,fit_count=len(features['train'])))
    print((out/'manifest.json').read_text(),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--features',required=True);p.add_argument('--out',required=True)
    p.add_argument('--kind',choices=['morph','regional','joint'],required=True)
    main(p.parse_args())
