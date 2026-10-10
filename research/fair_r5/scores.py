"""Per-map latent diagnostics; all reference moments come from normal validation."""
import numpy as np
from scipy.special import ndtr


def fit_moments(z):
    z=np.asarray(z,np.float64)
    return dict(mean=z.mean(0),std=np.maximum(z.std(0,ddof=1),1e-6))


def score(z,moments):
    z=np.asarray(z,np.float64)
    if z.ndim!=2 or z.shape[1]!=72 or not np.isfinite(z).all():
        raise ValueError('Expected finite N by 72 latents')
    s=(z-moments['mean'])/moments['std'];q=s*s
    top=np.sort(q,axis=1)[:,::-1];total=q.sum(1);dim=z.shape[1]
    p=np.sort(np.clip(2*ndtr(-abs(s)),1e-300,1-1e-12),axis=1)[:,:dim//2]
    t=np.arange(1,dim//2+1,dtype=np.float64)/dim
    dev=dim*(t*np.log(t/p)+(1-t)*np.log((1-t)/(1-p)))
    bj=np.maximum(np.where(p<t,dev,0).max(1),0)
    result=dict(raw_energy=(z*z).sum(1),diagonal_energy=total,raw_max=(z*z).max(1),
                diagonal_max=top[:,0],diagonal_top4=top[:,:4].mean(1),diagonal_bj=bj,
                angular_max=dim*top[:,0]/np.maximum(total,1e-12))
    if not all(np.isfinite(v).all() for v in result.values()):raise ValueError('Nonfinite statistic')
    return result
