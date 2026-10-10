"""Independent KDE reference and correlation-only anomaly control."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import gaussian_kde
from joint_density import gaussian_fit,mahalanobis,kde_fit,kde_logpdf
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'fair_r2'))
from common import low_fpr_metric


def main(a):
    rng=np.random.default_rng(77001);cov=np.array([[1,.9],[.9,1.]])
    reference=rng.multivariate_normal([0,0],cov,size=512)
    normal=rng.multivariate_normal([0,0],cov,size=4000)
    alternative=rng.multivariate_normal([0,0],cov*np.array([[1,-1],[-1,1]]),size=4000)
    checks={}
    for independent in (False,True):
        m=kde_fit(reference,independent=independent,shrink=0)
        got=kde_logpdf(normal[:31],m)
        if independent:
            expected=sum(gaussian_kde(reference[:,i],bw_method=m['bandwidth']).logpdf(normal[:31,i]) for i in range(2))
        else:expected=gaussian_kde(reference.T,bw_method=m['bandwidth']).logpdf(normal[:31].T)
        error=float(np.max(abs(got-expected)));assert error<1e-10
        checks['independent' if independent else 'joint']=error
        for part in (kde_logpdf(normal[:31][::-1],m)[::-1],
                     np.r_[kde_logpdf(normal[:1],m),kde_logpdf(normal[1:13],m),kde_logpdf(normal[13:31],m)]):
            assert np.allclose(got,part,rtol=1e-12,atol=1e-10)
        assert kde_logpdf(normal[:0],m).shape==(0,)
    scores={}
    for name,diagonal in [('diagonal',True),('joint',False)]:
        m=gaussian_fit(reference,diagonal=diagonal)
        scores[name]=low_fpr_metric(mahalanobis(normal,m),mahalanobis(alternative,m))
    assert scores['joint']>scores['diagonal']+.2
    out=dict(kind='GAUSSIAN_NUMERICAL_CONTROL_NOT_COSMOLOGICAL_OOD',scipy_log_density_errors=checks,
             pointwise_tolerance=1e-10,same_marginals_opposite_correlation_metric=scores)
    Path(a.out).write_text(json.dumps(out,indent=2));print(json.dumps(out))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args())
