"""Strong research starter: calibrated heterogenous detector ensemble.

Components (trained only on InD): JAX supervised summary encoder, JAX spectral
features, JAX multiscale/scattering-like statistics + shrinkage density.
Fits each separately; no synthetic OoD labels used. Calibration uses InD val.

The extra compute is intentional and MUST be included in budget comparisons.
This is not a certified FAIR SOTA solution.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
from candidates import hybrid_physics_jax as H
from candidates import multiscale_jax as M
from candidates import spectral_jax as S

NAMES=('hybrid_robust','hybrid_spectral','multiscale_cond','multiscale_neighbor','spectral')


def _components(states,maps,mask,cfg):
    h=H.all_scores(states['h'],maps,mask,cfg)
    m=M.all_scores(states['m'],maps,mask,cfg)
    s=S.predict(states['s'],maps,mask,cfg)
    return np.column_stack([h['hybrid_robust'],h['spectral'],
                            m['conditional'],m['neighbor'],s])


def fit(train,val,config,out_dir):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    info={}
    for k,mod in [('h',H),('m',M),('s',S)]:
        info[k]=mod.fit(train,val,config,out/k)
    states={'h':H.load_model(out/'h',config),'m':M.load_model(out/'m',config),
            's':S.load_model(out/'s',config)}
    calibration=[]
    for batch in val.iter_batches(batch_size=int(config['batch_size']),epochs=1,
                                  shuffle=True,seed=int(config['seed'])+454,
                                  limit=min(len(val),int(config.get('calibration_max',2048)))):
        calibration.append(_components(states,batch['maps'],batch['mask'],config))
    c=np.concatenate(calibration)
    np.save(out/'ensemble_calibration.npy',np.sort(c,axis=0))
    return {'components':info,'n_calibration':len(c),'component_names':NAMES,
            'note':'Computation includes all three fitted branches'}


def load_model(out_dir,config):
    out=Path(out_dir)
    return {'h':H.load_model(out/'h',config),'m':M.load_model(out/'m',config),
            's':S.load_model(out/'s',config),
            'calibration':np.load(out/'ensemble_calibration.npy')}


def all_scores(model,maps,mask,config):
    raw=_components(model,maps,mask,config)
    cal=model['calibration'];n=len(cal)
    surprise=np.column_stack([-np.log((n+1-np.searchsorted(cal[:,i],raw[:,i],side='right')) /
                                     (n+1.)) for i in range(len(NAMES))])
    # Rank-calibrate components before weighted averaging. No proxy-label fitting.
    # Conservative weighting balances model specialization against robustness.
    defaults=(.35,.10,.25,.15,.15)
    diverse=surprise@np.asarray(defaults)
    safeguard= .45*surprise[:,0] + .25*surprise[:,2] + .20*surprise[:,3] + .10*surprise[:,4]
    physics_first=.15*surprise[:,0]+.30*surprise[:,1]+.30*surprise[:,2]+.25*surprise[:,4]
    median=np.median(surprise,axis=1)
    return {'diverse':diverse,'safeguard':safeguard,
            'physics_first':physics_first,'median':median,
            **{'component_'+name:surprise[:,i] for i,name in enumerate(NAMES)}}


def predict(model,maps,mask,config):
    mode=str(config.get('ensemble_mode','physics_first'))
    choices=all_scores(model,maps,mask,config)
    if mode not in choices:
        raise ValueError(f'Unknown ensemble_mode {mode}')
    return np.asarray(choices[mode],np.float64)
