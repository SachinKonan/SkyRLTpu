"""Normal-only data and scoring helpers for the second FAIR research round."""
import hashlib,json
from pathlib import Path
import numpy as np

SPLIT_NAMES=('train','val','calibration','audit')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def splits():
    order=np.random.default_rng(2026).permutation(256)
    return {name:np.array([c*256+r for c in range(101) for r in rows]) for name,rows in zip(SPLIT_NAMES,[order[:160],order[160:192],order[192:224],order[224:]])}

def load_features(root):
    root=Path(root)
    values={k:np.load(root/f'{k}_latent.npy',allow_pickle=False).astype(np.float64) for k in SPLIT_NAMES}
    for k,ids in splits().items():
        if len(values[k])!=len(ids) or not np.isfinite(values[k]).all():raise ValueError(f'Invalid feature cache: {k}')
    return values

def load_labels(raw):
    path=Path(raw)/'label_newrealization.npy'
    if hashlib.md5(path.read_bytes()).hexdigest()!='2b88e3ccf8c8e66b445ee649886586c7':raise ValueError('Wrong physical labels')
    return np.load(path,allow_pickle=False).reshape((-1,5))

def low_fpr_metric(normal,positive):
    scores=np.r_[normal,positive]; y=np.r_[np.zeros(len(normal)),np.ones(len(positive))]
    order=np.argsort(-scores,kind='stable');v=scores[order];y=y[order]
    ends=np.r_[np.flatnonzero(np.diff(v)),len(v)-1];tp=np.cumsum(y)[ends];fp=ends+1-tp
    return float(np.interp(np.geomspace(.001,.05,100),np.r_[0,fp/len(normal)],np.r_[0,tp/len(positive)]).mean())

def whitening(x,floor=1e-4):
    mean=x.mean(0);scale=np.maximum(x.std(0),1e-8);z=(x-mean)/scale
    cov=np.cov(z,rowvar=False);e,u=np.linalg.eigh(cov);e=np.maximum(e,floor*e.max())
    matrix=((u/np.sqrt(e))@u.T)/scale[:,None]
    return mean,matrix

def score_report(cal,audit):
    result={}
    for key in cal:
        thresholds=np.quantile(cal[key],[.95,.99,.999]);groups=(audit[key].reshape(101,32)>thresholds[1]).sum(1)
        result[key]=dict(thresholds=thresholds.tolist(),normal_audit_count=len(audit[key]),exceedances=[int((audit[key]>t).sum()) for t in thresholds],max_cosmology_count_at_1pct=int(groups.max()),per_cosmology_counts_at_1pct=groups.tolist())
    return result

def write_json(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False))
