"""Normal-only low-dimensional density models for frozen field residuals."""
import numpy as np
from scipy.special import logsumexp


def gaussian_fit(x,diagonal=False,shrink=.1):
    x=np.asarray(x,np.float64);mean=x.mean(0);cov=np.atleast_2d(np.cov(x,rowvar=False))
    diag=np.diag(np.maximum(np.diag(cov),1e-12))
    cov=diag if diagonal else (1-shrink)*cov+shrink*diag
    return dict(mean=mean,cov=cov,inverse=np.linalg.inv(cov))


def mahalanobis(x,model):
    z=np.asarray(x,np.float64)-model['mean']
    return np.einsum('ni,ij,nj->n',z,model['inverse'],z)


def kde_fit(x,factor=1.,independent=False,shrink=.1):
    x=np.asarray(x,np.float64)
    if x.ndim!=2 or len(x)<3 or not np.isfinite(x).all():raise ValueError('Invalid normal reference')
    dimension=1 if independent else x.shape[1]
    h=float(factor*len(x)**(-1/(dimension+4)))
    g=gaussian_fit(x,diagonal=independent,shrink=shrink)
    covariance=g['cov']*h*h
    return dict(reference=x,covariance=covariance,inverse=np.linalg.inv(covariance),
                logdet=float(np.linalg.slogdet(covariance)[1]),independent=independent,
                factor=factor,bandwidth=h)


def kde_logpdf(x,model,batch=128):
    x=np.asarray(x,np.float64);reference=model['reference'];out=[]
    if x.ndim!=2 or x.shape[1]!=reference.shape[1] or not np.isfinite(x).all():
        raise ValueError('Invalid candidate statistics')
    if not len(x):return np.empty(0,np.float64)
    for lo in range(0,len(x),batch):
        difference=x[lo:lo+batch,None,:]-reference[None,:,:]
        if model['independent']:
            variance=np.diag(model['covariance'])
            value=(logsumexp(-.5*difference*difference/variance,axis=1)-
                   np.log(len(reference))-.5*np.log(2*np.pi*variance)).sum(1)
        else:
            distance=np.einsum('nri,ij,nrj->nr',difference,model['inverse'],difference)
            value=(logsumexp(-.5*distance,axis=1)-np.log(len(reference))-
                   .5*(x.shape[1]*np.log(2*np.pi)+model['logdet']))
        out.append(value)
    result=np.concatenate(out)
    if not np.isfinite(result).all():raise ValueError('Invalid joint density')
    return result


def choose_bandwidth(reference,folds,independent):
    candidates=[]
    for factor in (.5,1.,2.):
        oof=np.empty(len(reference))
        for fold in np.unique(folds):
            train=folds!=fold
            m=kde_fit(reference[train],factor=factor,independent=independent)
            oof[~train]=kde_logpdf(reference[~train],m)
        candidates.append(dict(factor=factor,normal_cv_nll=float(-oof.mean())))
    selected=min(candidates,key=lambda x:x['normal_cv_nll'])['factor']
    return kde_fit(reference,factor=selected,independent=independent),candidates
