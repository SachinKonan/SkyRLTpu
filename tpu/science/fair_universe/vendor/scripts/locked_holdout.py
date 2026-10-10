"""ONE-SHOT frozen evaluation on fresh synthetic holdout.

The official FAIR OoD labels are hidden. This holdout does NOT establish a
competition leaderboard score. Don't use the outcome to tune the code/weights.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter
from fairdiscover.data import MapStore, stage
from fairdiscover.config import read_config
from fairdiscover.shifts import proxy_shift, FAMILIES
from fairdiscover.metric import per_family_scores
from fairdiscover.worker import load_candidate
from scripts.run_proxy_comparison import make_fields, extra_shift

ROOT=Path(__file__).resolve().parents[1]
NOVEL=('anisotropic_smoothing','localized_peak','phase_randomization')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--checkpoint-root',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--seed',type=int,default=965205)
    p.add_argument('--only',default='spectral,autoencoder,multiscale,hybrid_v2,diverse_v3')
    p.add_argument('--infer-batch-size',type=int,default=128)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    data=out/'raw';data.mkdir(parents=True,exist_ok=True)
    manifest=make_fields(data,int(a.seed),64)
    staged=out/'staged'
    stage(str(data),str(staged),mode='sample',seed=int(a.seed)+1)
    store=MapStore(staged)
    ids=np.asarray([(c,r) for c in range(store.maps.shape[0]) for r in range(store.maps.shape[1])],np.int32)
    cfg=read_config(hybrid_mode='hybrid_robust',ensemble_mode='physics_first',evolved_mode='conditional')
    trained=Path(a.checkpoint_root)
    definitions=[('spectral','spectral_jax.py'),('autoencoder','baseline_jax.py'),
                 ('multiscale','multiscale_jax.py'),('hybrid_v2','hybrid_physics_jax.py'),
                 ('diverse_v3','diverse_ensemble_jax.py')]
    models={}
    for name,filename in definitions:
        checkpoint={'hybrid_v2':'hybrid','diverse_v3':'diverse'}.get(name,name)
        module=load_candidate(ROOT/'candidates'/filename)
        if name in a.only.split(','):
            models[name]=(module,module.load_model(trained/checkpoint,cfg))
    families=(*FAMILIES,*NOVEL)
    scores={name:{k:[] for k in ('ind',*families)} for name in models}
    rng=np.random.default_rng(int(a.seed)+450)
    for st in range(0,len(ids),a.infer_batch_size):
        ii=ids[st:st+a.infer_batch_size]
        clean=store.get_clean(ii)
        kinds={'ind':store.apply_noise(clean,rng,cfg)}
        for fi,fam in enumerate(families):
            modified=[]
            for x,pair in zip(clean,ii):
                r=np.random.default_rng(np.random.SeedSequence([int(a.seed),fi,int(pair[0]),int(pair[1])]))
                if fam in FAMILIES:
                    modified.append(proxy_shift(x,store.mask,fam,r,1.0))
                else:
                    modified.append(extra_shift(x,store.mask,fam,r))
            kinds[fam]=store.apply_noise(np.stack(modified),rng,cfg)
        for name,(module,state) in models.items():
            for fam,xx in kinds.items():
                values=np.asarray(module.predict(state,xx,store.mask,cfg),np.float64)
                if values.shape != (len(ii),) or not np.isfinite(values).all():
                    raise ValueError(f'{name} gave invalid output for {fam}')
                scores[name][fam].extend(values.tolist())
        print('holdout',st+len(ii),'/',len(ids),flush=True)
    result={}
    for name,ss in scores.items():
        family=per_family_scores(ss['ind'],{f:ss[f] for f in families})
        result[name]={'development_shift_mean':float(np.mean([family[f] for f in FAMILIES])),
                      'unseen_shift_mean':float(np.mean([family[f] for f in NOVEL])),
                      'all_shift_mean':float(np.mean(list(family.values()))),
                      'per_family':family}
    meta={'kind':'One-shot synthetic holdout, NOT FAIR', 'seed':a.seed,
          'frozen_selected':'diverse_v3/physics_first',
          'source_manifest':manifest, 'n_ind':len(ids), 'results':result}
    (out/'holdout_result.json').write_text(json.dumps(meta,indent=2))
    print('RESULT\n'+json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()
