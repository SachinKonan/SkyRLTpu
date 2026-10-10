"""Numerical controls on a known Gaussian null, not cosmological performance."""
import argparse
import json
from pathlib import Path
import numpy as np
from scipy.special import ndtri
from scipy.stats import chi2
from scores import score


def main(a):
    rng=np.random.default_rng(12814);z=rng.normal(size=(50000,72))
    moments=dict(mean=np.zeros(72),std=np.ones(72));v=score(z,moments)
    thresholds=dict(raw_energy=float(chi2.ppf(.99,72)),raw_max=float(ndtri((1+.99**(1/72))/2)**2))
    rates={k:float(np.mean(v[k]>t)) for k,t in thresholds.items()}
    for r in rates.values():assert abs(r-.01)<5*np.sqrt(.01*.99/len(z))
    assert abs(v['raw_energy'].mean()-72)<.2 and abs(v['raw_energy'].std()-12)<.2
    assert np.allclose(v['raw_energy'],v['diagonal_energy'])
    assert np.allclose(v['raw_max'],v['diagonal_max'])
    assert all(np.array_equal(value[:31],score(z[:31][::-1],moments)[k][::-1]) for k,value in v.items())
    assert np.allclose(score(3*z[:31],moments)['angular_max'],v['angular_max'][:31],rtol=1e-12,atol=1e-12)
    assert np.all(score(1.2*z[:31],moments)['diagonal_bj']>=v['diagonal_bj'][:31]-1e-12)
    report=dict(kind='GAUSSIAN_NUMERICAL_CONTROL_NOT_COSMOLOGICAL_OOD',count=len(z),
                analytical_1pct_thresholds=thresholds,observed_fractions=rates,
                energy_mean=float(v['raw_energy'].mean()),energy_std=float(v['raw_energy'].std()),
                order_invariance=True,angular_scale_invariance=True,tail_statistic_monotonicity=True)
    Path(a.out).write_text(json.dumps(report,indent=2));print(json.dumps(report))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);main(p.parse_args())
