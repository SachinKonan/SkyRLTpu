"""Normal generalization and labeled synthetic stress tests, NOT official OOD accuracy."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import rankdata
from score import make_encoder,nll
from train import load_training,NOISE

DEVELOPMENT=('spectral_low','spectral_high','quadratic_peak','gaussian_mixture')
NOVEL=('anisotropic','spatial_modulation')


def shift(clean,mask,family,strength,reference=None):
    x=clean*mask; obs=x[:,mask]; mean=obs.mean(1); sd=obs.std(1)
    def match(y):
        yy=y[:,mask]; sm=yy.std(1)
        return ((y-yy.mean(1)[:,None,None])* (sd/np.maximum(sm,1e-8))[:,None,None]+mean[:,None,None])*mask
    def blur(sigma):
        norm=gaussian_filter(mask.astype(float),sigma=sigma)
        return gaussian_filter(x,sigma=(0,*sigma))/np.maximum(norm,.1)
    if family=='spectral_low':return ((1-strength)*x+strength*blur((2.,2.)))*mask
    if family=='spectral_high':return (x+strength*(x-blur((2.,2.))))*mask
    if family=='quadratic_peak':
        u=(x-mean[:,None,None])/np.maximum(sd[:,None,None],1e-8)
        return match(x+strength*.3*sd[:,None,None]*(u*u-1))
    if family=='gaussian_mixture':
        if reference is None:raise ValueError('Requires same-cosmology independent reference')
        return match((1-strength)*x+strength*reference)
    if family=='anisotropic':return match((1-strength)*x+strength*blur((.5,3.)))
    if family=='spatial_modulation':
        wave=np.sin(np.arange(mask.shape[0])*2*np.pi/mask.shape[0])[:,None]
        return match(x*(1+strength*.5*wave))
    raise ValueError(family)


def low_fpr_metric(normal,anomaly):
    # Same ROC interpolation convention as sklearn.roc_curve, including duplicate FPRs.
    scores=np.r_[normal,anomaly]; labels=np.r_[np.zeros(len(normal)),np.ones(len(anomaly))]
    order=np.argsort(-scores,kind='stable'); s=scores[order]; y=labels[order]
    ends=np.r_[np.flatnonzero(np.diff(s)),len(s)-1]
    tp=np.cumsum(y)[ends]; fp=(ends+1)-tp
    tpr=np.r_[0,tp/len(anomaly)];fpr=np.r_[0,fp/len(normal)]
    return float(np.interp(np.geomspace(.001,.05,100),fpr,tpr).mean())


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--run',required=True);p.add_argument('--count',type=int,default=512);p.add_argument('--batch',type=int,default=16);p.add_argument('--novel',action='store_true');a=p.parse_args()
    run=Path(a.run);maps,mask,labels,splits=load_training(a.raw);enc=make_encoder(run,mask,a.batch)
    with np.load(run/'density/model.npz',allow_pickle=False) as f:d={k:f[k] for k in f.files}
    cal=np.load(run/'density/calibration_scores.npy'); q10,mid,q90=np.quantile(cal,[.1,.5,.9])
    def two_tail(s):return np.where(s>=mid,(s-mid)/max(q90-mid,1e-8),(mid-s)/max(mid-q10,1e-8))
    ids=np.random.default_rng(475 if a.novel else 474).permutation(splits['audit' if a.novel else 'val'])[:a.count]
    families=NOVEL if a.novel else DEVELOPMENT
    outputs={}; normal=[]; second=[]
    for i in range(0,len(ids),a.batch):
        idx=ids[i:i+a.batch];x=np.zeros((len(idx),*mask.shape),np.float32);x[:,mask]=maps[idx]
        # Reference maps drawn ONLY from fitting data, same cosmology. No test inputs.
        refidx=np.array([splits['train'][160*(n//256)+(n%160)] for n in idx])
        ref=np.zeros_like(x);ref[:,mask]=maps[refidx]
        rng=np.random.default_rng(7300+i+int(a.novel));noise=rng.normal(0,NOISE,x.shape).astype(np.float32)*mask
        normal.extend(nll(enc(x+noise),d));second.extend(nll(enc(x+rng.normal(0,NOISE,x.shape).astype(np.float32)*mask),d))
        for family in families:
            for strength in (.2,.5):
                key=f'{family}:{strength}'
                shifted=shift(x,mask,family,strength,ref)
                outputs.setdefault(key,[]).extend(nll(enc(shifted+noise),d))
    normal=np.asarray(normal);second=np.asarray(second)
    metric=[]
    for name,v in outputs.items():
        metric.append(dict(family=name,one_sided=low_fpr_metric(cal,np.asarray(v)),two_sided=low_fpr_metric(two_tail(cal),two_tail(np.asarray(v)))))
    thresholds=np.quantile(cal,[.95,.99,.999])
    report=dict(kind='synthetic_stress_NOT_OFFICIAL_OOD',phase='novel_once' if a.novel else 'development',normal_count=len(normal),reference_count=len(cal),noise_rank_correlation=float(np.corrcoef(rankdata(normal),rankdata(second))[0,1]),normal_false_positive_counts=[int((normal>t).sum()) for t in thresholds],null_low_fpr_metric=low_fpr_metric(cal,normal),families=metric)
    tag='novel' if a.novel else 'development'
    np.savez(run/f'{tag}_diagnostic_scores.npz',normal=normal,second=second,**{k:np.asarray(v) for k,v in outputs.items()})
    (run/f'{tag}_diagnostics.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
