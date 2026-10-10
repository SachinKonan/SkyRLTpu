"""Cosmology-conditioned covariance and observable conditional-likelihood controls."""
import argparse,json,time
from pathlib import Path
import numpy as np
from scipy.special import logsumexp
from common import load_features,load_labels,splits,write_json,sha,score_report,low_fpr_metric

def component_statistics(x,means,precision,logdet):
    q=np.empty((len(x),len(means)))
    for k in range(len(means)):
        d=x-means[k];q[:,k]=np.einsum('ni,ij,nj->n',d,precision[k],d,optimize=True)
    return q,.5*(q+logdet[None]+x.shape[1]*np.log(2*np.pi))

def predictor(run):
    with np.load(Path(run)/'model.npz') as f:m={k:f[k] for k in f.files}
    def predict(features):
        x=(features-m['mean'])/m['scale'];q,n=component_statistics(x,m['centers'],m['precision'],m['logdet'])
        pq,pn=component_statistics(x[:,:20],m['centers'][:,:20],m['power_precision'],m['power_logdet'])
        joint=-logsumexp(-n,axis=1)+np.log(n.shape[1]);power=-logsumexp(-pn,axis=1)+np.log(pn.shape[1])
        w=np.exp(-pn-logsumexp(-pn,axis=1,keepdims=True))
        return dict(joint_nll=joint,profile_nll=n.min(1),minimum_mahalanobis=q.min(1),power_nll=power,conditional_shape_nll=joint-power,posterior_conditional_energy=np.sum(w*np.maximum(q-pq,0),axis=1))
    return predict

def fit(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);features=load_features(a.features);labels=load_labels(a.raw);start=time.monotonic()
    mean=features['train'].mean(0);scale=np.maximum(features['train'].std(0),1e-8)
    x=(features['train']-mean)/scale;val=(features['val']-mean)/scale
    n,d=x.shape;x=x.reshape(101,160,d);centers=x.mean(1);residual=x-centers[:,None];cov=np.einsum('cni,cnj->cij',residual,residual)/159;pooled=cov.mean(0)
    candidates=[];best=np.inf;selected=None
    for shrink in (.25,.5,.75,1.):
        for ridge in (.001,.01):
            v=(1-shrink)*cov+shrink*pooled[None]+ridge*np.eye(d)[None]
            sign,logdet=np.linalg.slogdet(v)
            if not (sign>0).all():raise ValueError('Covariance is not positive definite')
            precision=np.linalg.inv(v);z=val.reshape(101,32,d)-centers[:,None]
            score=.5*(np.einsum('cni,cij,cnj->cn',z,precision,z,optimize=True)+logdet[:,None]+d*np.log(2*np.pi))
            record=dict(shrink=shrink,ridge=ridge,val_true_cosmology_nll=float(score.mean()));candidates.append(record);print(json.dumps(record),flush=True)
            if score.mean()<best:best=score.mean();selected=(v,precision,logdet,record)
    cov,precision,logdet,choice=selected;pv=cov[:,:20,:20];power_precision=np.linalg.inv(pv);_,power_logdet=np.linalg.slogdet(pv)
    np.savez(out/'model.npz',mean=mean,scale=scale,centers=centers,precision=precision,logdet=logdet,power_precision=power_precision,power_logdet=power_logdet,cosmology=labels.reshape(101,256,5)[:,0,:2])
    write_json(out/'fit.json',dict(selected=choice,candidates=candidates,normal_feature_hashes={k:sha(Path(a.features)/f'{k}_latent.npy') for k in features},seconds=time.monotonic()-start))
    (out/'source.py').write_bytes(Path(__file__).read_bytes());evaluate(a)

def evaluate(a):
    out=Path(a.out);predict=predictor(out);features=load_features(a.features)
    cal=predict(features['calibration']);audit=predict(features['audit']);np.savez(out/'calibration_scores.npz',**cal);np.savez(out/'audit_scores.npz',**audit)
    report=dict(kind='normal_and_synthetic_NOT_OFFICIAL_OOD',normal=score_report(cal,audit),probes={})
    if a.probes:
        for path in Path(a.probes).glob('*.npz'):
            with np.load(path) as f:
                all_scores={};metrics={}
                for key in f.files:
                    if key=='ids':continue
                    values=predict(f[key]);metrics[key]={name:low_fpr_metric(cal[name],v) for name,v in values.items()};all_scores.update({key+'__'+name:v for name,v in values.items()})
            np.savez(out/(path.stem+'_scores.npz'),**all_scores);report['probes'][path.stem]=metrics
    report.update(model_sha256=sha(out/'model.npz'),runtime_sha256=sha(__file__));write_json(out/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','evaluate']);p.add_argument('--features',required=True);p.add_argument('--raw',required=True);p.add_argument('--out',required=True);p.add_argument('--probes');a=p.parse_args()
    fit(a) if a.mode=='fit' else evaluate(a)
