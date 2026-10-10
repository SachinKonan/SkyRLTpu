"""Fixed development-only ablations of FAIR v2 candidate.

Never consults a separate holdout. All models receive the SAME noisy maps,
synthetic shifts and score metric. This does not estimate FAIR competition score.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from fairdiscover.data import MapStore, DataView
from fairdiscover.shifts import FAMILIES, proxy_shift
from fairdiscover.metric import per_family_scores
from fairdiscover.worker import load_candidate
from fairdiscover.config import read_config

ROOT=Path(__file__).resolve().parents[1]


def evaluate(stage, cfg, models, sample_cap=1024, seed=1946):
    store=MapStore(stage)
    ids=store.ids('reward')
    rng=np.random.default_rng(seed)
    if sample_cap and len(ids)>sample_cap:
        ids=ids[np.sort(rng.choice(len(ids),sample_cap,replace=False))]
    labels={name:{k:[] for k in ('ind',)+FAMILIES} for name in models}
    for st in range(0,len(ids),int(cfg['infer_batch_size'])):
        subset=ids[st:st+int(cfg['infer_batch_size'])]
        x=store.get_clean(subset)
        inputs={'ind':store.apply_noise(x,rng,cfg)}
        for fi,fam in enumerate(FAMILIES):
            shift=np.stack([proxy_shift(img,store.mask,fam,
                   np.random.default_rng(np.random.SeedSequence([
                       seed,100+fi,int(pair[0]),int(pair[1])])),1.0)
                   for img,pair in zip(x,subset)])
            inputs[fam]=store.apply_noise(shift,rng,cfg)
        for name,(mod,state) in models.items():
            for fam,arr in inputs.items():
                if hasattr(mod,'all_scores'):
                    result=mod.all_scores(state,arr,store.mask,cfg)
                    for subname,values in result.items():
                        key=name+'/'+subname
                        labels.setdefault(key,{k:[] for k in ('ind',)+FAMILIES})[fam].extend(np.asarray(values).tolist())
                else:
                    labels[name][fam].extend(np.asarray(mod.predict(state,arr,store.mask,cfg)).tolist())
    report={}
    for name,kind in labels.items():
        if not kind['ind']: continue
        per=per_family_scores(kind['ind'],{fam:kind[fam] for fam in FAMILIES})
        report[name]={'mean':float(np.mean(list(per.values()))),'minimum':float(min(per.values())),'per_family':per}
    return report,len(ids)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--stage',required=True)
    p.add_argument('--out',required=True)
    p.add_argument('--steps',type=int,default=250)
    p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--reward-count',type=int,default=1024)
    p.add_argument('--seed',type=int,default=31415)
    p.add_argument('--only',default='autoencoder,spectral,multiscale,hybrid,diverse')
    p.add_argument('--infer-batch-size',type=int,default=128)
    a=p.parse_args()
    out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    cfg=read_config(steps=a.steps,batch_size=a.batch_size,infer_batch_size=a.infer_batch_size,seed=a.seed,hybrid_mode='hybrid_robust',evolved_mode='conditional')
    store=MapStore(a.stage)
    models={};fits={}
    for name,path in [('autoencoder','baseline_jax.py'),('spectral','spectral_jax.py'),
                      ('multiscale','multiscale_jax.py'),('hybrid','hybrid_physics_jax.py'),('diverse','diverse_ensemble_jax.py')]:
        if name not in a.only.split(','): continue
        mod=load_candidate(ROOT/'candidates'/path)
        work=out/'models'/name;work.mkdir(parents=True,exist_ok=True)
        st=time.perf_counter()
        info=mod.fit(DataView(store,'train',cfg),DataView(store,'val',cfg),cfg,work)
        print('FIT',name,info,'elapsed',time.perf_counter()-st,flush=True)
        models[name]=(mod,mod.load_model(work,cfg))
        fits[name]={'fit':info,'seconds':time.perf_counter()-st}
    results,n=evaluate(a.stage,cfg,models,a.reward_count)
    ranking=sorted(results.items(),key=lambda k:k[1]['mean'],reverse=True)
    bundle={'kind':'development proxy / synthetic anomalies, not FAIR evaluation',
            'data_stage':str(Path(a.stage).resolve()),'steps':a.steps,'batch_size':a.batch_size,
            'in_distribution_count':n,'fits':fits,'ranked':ranking}
    (out/'development_comparison.json').write_text(json.dumps(bundle,indent=2))
    for k,v in ranking:
        print(f'{k:26s} mean={v["mean"]:.5f} worst={v["minimum"]:.5f}',flush=True)
    print('SAVED',out/'development_comparison.json')

if __name__=='__main__':main()
