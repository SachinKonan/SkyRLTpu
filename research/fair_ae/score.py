"""Fit frozen train-only latent densities, then independently score public maps."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import jax
import jax.numpy as jnp
from flax import serialization
from scipy.cluster.vq import kmeans2
from scipy.special import logsumexp
from train import make_model, load_training, NOISE

TEST='WIDE12H_bin2_2arcmin_kappa_test_phase2_new_v2.npy'


def make_encoder(run, mask, batch=16):
    run=Path(run); cfg=json.loads((run/'config.json').read_text()); norm=json.loads((run/'normalization.json').read_text())
    if cfg.get('architecture')=='physics':
        from features import make_predict
        return make_predict(mask,batch)
    if cfg.get('architecture')=='summary':
        from summary_train import SummaryNet
        from features import make_predict
        features=make_predict(mask,batch); net=SummaryNet(cfg['latent'],cfg['dimensions'])
        params=jax.tree.map(jnp.asarray,serialization.msgpack_restore((run/'best.msgpack').read_bytes()))
        center=np.asarray(norm['input_center']); scale=np.asarray(norm['input_scale'])
        apply=jax.jit(lambda x:net.apply({'params':params},x))
        def embed(f):
            z,r,_=apply(jnp.asarray((f-center)/scale,jnp.float32))
            z=np.asarray(z)
            if cfg.get('representation')=='latent_and_residual':
                observed=(f-np.asarray(norm['target_center']))/np.asarray(norm['target_scale'])
                e=(observed-np.asarray(r))**2
                residual=np.column_stack([np.log1p(e[:,lo:hi].mean(1)) for lo,hi in [(0,20),(20,60),(60,72)]])
                z=np.column_stack([z,residual])
            return z
        def prediction(x):return embed(features(x))
        prediction.from_features=embed
        return prediction
    model=make_model(cfg); m=jnp.asarray(mask,jnp.float32)
    params=jax.tree.map(jnp.asarray,serialization.msgpack_restore((run/'best.msgpack').read_bytes()))
    @jax.jit
    def encode(x):
        x=((x-norm['mean'])/norm['input_scale'])*m
        z,rec,theta=model.apply({'params':params},x,m)
        return z
    def predict(x):
        result=[]
        for i in range(0,len(x),batch):
            part=x[i:i+batch]; n=len(part)
            if n<batch:part=np.pad(part,((0,batch-n),(0,0),(0,0)))
            result.append(np.asarray(encode(part))[:n])
        return np.concatenate(result)
    return predict


def nll(z,d):
    w=(z-d['mean'])@d['whiten']
    delta=w[:,None,:]-d['centers'][None]
    squared=np.sum(delta**2/d['variances'][None],axis=-1)
    if str(d.get('score_mode','likelihood'))=='nearest_standardized_distance':
        return squared.min(axis=1)
    logp=-.5*(squared+np.log(d['variances']).sum(-1)[None]+w.shape[1]*np.log(2*np.pi))
    return -logsumexp(logp+np.log(d['weights'])[None],axis=1)


def fit_density(train,val,cosmology=None):
    mean=train.mean(0); scale=np.maximum(train.std(0),1e-6)
    standardized=(train-mean)/scale
    cov=np.cov(standardized,rowvar=False)
    eig,u=np.linalg.eigh(cov)
    # Shared train-fitted whitening for every K; eigenvalue floor avoids collapsed directions.
    eig=np.maximum(eig,.01*max(eig.max(),1e-12)); whiten=((u/np.sqrt(eig))@u.T)/scale[:,None]
    w=(train-mean)@whiten
    candidates=[]; models=[]
    choices=[('kmeans',k) for k in (1,4,16,64)]
    if cosmology is not None: choices.append(('known_cosmology',len(np.unique(cosmology,axis=0))))
    for kind,k in choices:
        if kind=='kmeans':
            centers,labels=kmeans2(w,k,iter=60,minit='++',seed=2026,missing='raise')
        else:
            _,labels=np.unique(cosmology,axis=0,return_inverse=True)
            centers=np.stack([w[labels==c].mean(0) for c in range(k)])
        counts=np.bincount(labels,minlength=k)
        # Shrink component variances toward shared unit-scale coordinates.
        variances=[]
        for c in range(k):
            residual=w[labels==c]-centers[c]
            variances.append((np.square(residual).sum(0)+32)/(len(residual)+32))
        d=dict(mean=mean,whiten=whiten,centers=centers,variances=np.maximum(variances,.05),weights=(counts+1)/(len(w)+k))
        scores=nll(val,d)
        rec=dict(kind=kind,k=k,val_mean_nll=float(scores.mean()),min_cluster_count=int(counts.min()),val_std_nll=float(scores.std()))
        print('DENSITY',json.dumps(rec),flush=True); candidates.append(rec); models.append(d)
    best=int(np.argmin([r['val_mean_nll'] for r in candidates]))
    return models[best],dict(candidates=candidates,selected_k=candidates[best]['k'],selected_kind=candidates[best]['kind'])


def fit(args):
    run=Path(args.run); dest=run/'density'
    dest.mkdir(exist_ok=False)
    maps,mask,labels,splits=load_training(args.raw)
    encode=make_encoder(run,mask,args.batch)
    zs={}
    for name,ids in splits.items():
        rng=np.random.default_rng(300+list(splits).index(name)); chunks=[]
        cfg=json.loads((run/'config.json').read_text())
        cached=Path(cfg.get('source_features','/nonexistent'))/(name+'_latent.npy')
        if hasattr(encode,'from_features') and cached.exists():
            feats=np.load(cached,allow_pickle=False)
            if len(feats)!=len(ids):raise ValueError('Cached normal feature count mismatch')
            zs[name]=np.concatenate([encode.from_features(feats[i:i+256]) for i in range(0,len(feats),256)])
            np.save(dest/(name+'_latent.npy'),zs[name])
            print('ENCODED_FROM_NORMAL_FEATURE_CACHE',name,zs[name].shape,flush=True)
            continue
        for i in range(0,len(ids),args.batch):
            selected=ids[i:i+args.batch]
            x=np.zeros((len(selected),*mask.shape),np.float32); x[:,mask]=maps[selected]
            x+=rng.normal(0,NOISE,x.shape).astype(np.float32)*mask
            chunks.append(encode(x))
        zs[name]=np.concatenate(chunks); np.save(dest/(name+'_latent.npy'),zs[name])
        print('ENCODED',name,zs[name].shape,flush=True)
    d,report=fit_density(zs['train'],zs['val'],np.round(labels[splits['train'],:2],6))
    np.savez(dest/'model.npz',**d)
    for name in ('calibration','audit'):
        s=nll(zs[name],d)
        report[name]=dict(count=len(s),mean=float(s.mean()),std=float(s.std()),quantiles=np.quantile(s,[.5,.9,.95,.99,.999]).tolist())
        np.save(dest/(name+'_scores.npy'),s)
    cal=nll(zs['calibration'],d); audit=nll(zs['audit'],d)
    thresholds=np.quantile(cal,[.95,.99,.999])
    theta=labels[splits['audit'],:2]
    _,groups=np.unique(np.round(theta,6),axis=0,return_inverse=True)
    report['normal_audit_by_cosmology']=[dict(theta=theta[groups==g][0].tolist(),count=int((groups==g).sum()),exceedance_counts=[int((audit[groups==g]>t).sum()) for t in thresholds]) for g in np.unique(groups)]
    report['normal_audit_thresholds']=thresholds.tolist()
    report['method']='Train-whitened diagonal Gaussian mixture; K-means or known cosmology centers selected by normal validation NLL; no official-test fitting'
    report['runtime_files']={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['train.py','features.py','summary_train.py']}
    report['frozen_files']={name:hashlib.sha256((run/name).read_bytes()).hexdigest() for name in ['config.json','normalization.json','best.msgpack','source.py','density/model.npz']}
    report['checkpoint_sha256']=hashlib.sha256((run/'best.msgpack').read_bytes()).hexdigest()
    report['score_source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (dest/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)


def export(args):
    run=Path(args.run); test=Path(args.test)
    if test.name!=TEST:raise ValueError('Requires current official Phase-2 test filename')
    report=json.loads((run/'density/report.json').read_text())
    if hashlib.sha256((run/'best.msgpack').read_bytes()).hexdigest()!=report['checkpoint_sha256']:raise ValueError('Checkpoint changed')
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest()!=report['score_source_sha256']:raise ValueError('Scoring source changed after fitting')
    for name,expected in report['runtime_files'].items():
        if hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()!=expected:raise ValueError(f'Runtime changed: {name}')
    for name,expected in report['frozen_files'].items():
        if hashlib.sha256((run/name).read_bytes()).hexdigest()!=expected:raise ValueError(f'Frozen artifact changed: {name}')
    maskfile=Path(args.raw)/'WIDE12H_bin2_2arcmin_mask.npy'
    if hashlib.md5(maskfile.read_bytes()).hexdigest()!='29b8bbb8a9ad41d3dacfaf61008a8268':raise ValueError('Mask checksum mismatch')
    mask=np.load(maskfile,allow_pickle=False)
    x=np.load(test,mmap_mode='r',allow_pickle=False)
    if len(x)!=10000:raise ValueError('Expected exactly 10000 test maps')
    encode=make_encoder(run,mask,args.batch)
    with np.load(run/'density/model.npz',allow_pickle=False) as z:d={k:z[k] for k in z.files}
    def reconstruct(part):
        part=np.asarray(part,np.float32)
        if part.shape[1:]==mask.shape:return part*mask
        if part.shape[1:]==(int(mask.sum()),):
            a=np.zeros((len(part),*mask.shape),np.float32); a[:,mask]=part; return a
        if part.shape[1:]==(mask.size,):return part.reshape((-1,*mask.shape))*mask
        raise ValueError(f'Unexpected test shape {part.shape}')
    # Check fixed function on actual maps; pass/fail only, never modify based on test.
    probe=reconstruct(x[:8]); z=encode(probe)
    alone=encode(probe[:1]); reverse=encode(probe[::-1])[::-1]
    if not np.allclose(z[:1],alone,rtol=1e-5,atol=1e-6) or not np.allclose(z,reverse,rtol=1e-5,atol=1e-6):raise ValueError('Test-sample independence failed')
    scores=[]
    for i in range(0,len(x),args.batch):scores.extend(nll(encode(reconstruct(x[i:i+args.batch])),d).tolist())
    if len(scores)!=10000 or not np.isfinite(scores).all():raise ValueError('Invalid scores')
    out=Path(args.out); out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('result.json',json.dumps({'ood_scores':scores},allow_nan=False))
    with zipfile.ZipFile(out) as z:
        assert z.namelist()==['result.json'] and len(json.loads(z.read('result.json'))['ood_scores'])==10000
    print(json.dumps(dict(archive=str(out),count=len(scores),sha256=hashlib.sha256(out.read_bytes()).hexdigest(),independence_verified=True)),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','export']);p.add_argument('--raw',required=True);p.add_argument('--run',required=True);p.add_argument('--batch',type=int,default=16);p.add_argument('--test');p.add_argument('--out');a=p.parse_args()
    fit(a) if a.mode=='fit' else export(a)
