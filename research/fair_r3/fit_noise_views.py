"""Controlled R2 density ablation: same transforms/hyperparameters, 4 noise views.

Whitening and all normalization are kept equal to the one-view baseline. Only
the fitting examples change; validation, calibration and audit remain fixed.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import serialization
import optax
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_r2'))
from flow import AffineFlow,prepare,evaluate
from common import load_features,load_labels,splits,sha,write_json


def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    features=load_features(a.features);labels=load_labels(a.raw)
    z,c,jac,norm=prepare(features,labels,a.condition,a.floor,a.transform)
    norm['cosmology_grid']=labels.reshape(101,256,5)[:,0,:2]
    views=Path(a.views);meta=json.loads((views/'manifest.json').read_text())
    assert meta['kind']=='FITTING_REALIZATIONS_ONLY_NEW_NOISE_VIEWS'
    assert sha(views/'training_views.npy')==meta['views_sha256']
    assert np.array_equal(np.load(views/'ids.npy'),splits()['train'])
    extra=np.load(views/'training_views.npy').astype(np.float64)
    assert extra.shape==(3,len(z['train']),72)
    raw=extra.reshape(-1,72);u=(raw-norm['target_offset'])/norm['target_scale']
    aug=((np.arcsinh(u)-norm['mean'])@norm['whiten']).astype(np.float32)
    aug_jac=np.linalg.slogdet(norm['whiten'])[1]-np.sum(np.log(norm['target_scale'])+.5*np.log1p(u*u),axis=1)
    z['train']=np.concatenate([z['train'],aug]);c['train']=np.tile(c['train'],(4,1))
    jac['train']=np.concatenate([jac['train'],aug_jac])
    np.savez(out/'normalization.npz',**norm)
    cfg=vars(a).copy();cfg.update(dim=72,noise_views=4,training_examples=len(z['train']),
        distinct_simulator_realizations=len(splits()['train']),views_manifest_sha256=sha(views/'manifest.json'),
        baseline_feature_hashes={k:sha(Path(a.features)/f'{k}_latent.npy') for k in features})
    write_json(out/'config.json',cfg);(out/'source.py').write_bytes(Path(__file__).read_bytes())
    net=AffineFlow(72,a.layers,a.width,a.scale_bound)
    params=net.init(jax.random.PRNGKey(a.seed),j.zeros((a.batch,72)),j.zeros((a.batch,2)))['params']
    tx=optax.chain(optax.clip_by_global_norm(5),optax.adamw(optax.cosine_decay_schedule(3e-4,a.steps,alpha=.05),weight_decay=1e-4))
    opt=tx.init(params);xtrain=j.asarray(z['train']);ctrain=j.asarray(c['train'])
    xval=j.asarray(z['val']);cval=j.asarray(c['val'])
    def loss(p,x,condition):return net.apply({'params':p},x,condition)[0].mean()
    @jax.jit
    def update(p,o,ids):
        v,g=jax.value_and_grad(loss)(p,xtrain[ids],ctrain[ids]);u,o=tx.update(g,o,p)
        return optax.apply_updates(p,u),o,v
    validate=jax.jit(lambda p:loss(p,xval,cval));rng=np.random.default_rng(a.seed)
    best=np.inf;start=time.monotonic()
    for step in range(1,a.steps+1):
        params,opt,train=update(params,opt,rng.integers(len(xtrain),size=a.batch))
        if step==1 or step%250==0 or step==a.steps:
            value=float(validate(params))-float(jac['val'].mean())
            record=dict(step=step,train_nll=float(train)-float(jac['train'].mean()),
                        val_conditional_nll=value,seconds=time.monotonic()-start)
            if not np.isfinite(value):raise FloatingPointError('Nonfinite validation likelihood')
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print(json.dumps(record),flush=True)
            if value<best:
                best=value;(out/'best.msgpack').write_bytes(serialization.to_bytes(params));write_json(out/'best.json',record)
    write_json(out/'complete.json',dict(steps=a.steps,best_normal_val_nll=best,seconds=time.monotonic()-start,
                                      training_examples=len(z['train']),noise_views=4))
    evaluate(a)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('raw','features','views','out','probes'):p.add_argument('--'+name,required=True)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=10000)
    p.add_argument('--batch',type=int,default=256);p.add_argument('--layers',type=int,default=8)
    p.add_argument('--width',type=int,default=128);p.add_argument('--floor',type=float,default=1e-4)
    p.add_argument('--condition',default='cosmology',choices=['cosmology'])
    p.add_argument('--transform',default='asinh',choices=['asinh']);p.add_argument('--scale-bound',type=float,default=.5)
    main(p.parse_args())
