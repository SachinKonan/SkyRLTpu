"""Affine-coupling density models; all fitting and model selection are normal-only."""
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import linen as nn,serialization
import optax
from scipy.special import logsumexp
from common import load_features,load_labels,splits,whitening,write_json,sha,score_report,low_fpr_metric

jax.config.update('jax_default_matmul_precision','highest')

class AffineFlow(nn.Module):
    dim:int
    layers:int=8
    width:int=128
    scale_bound:float=.5
    @nn.compact
    def __call__(self,x,condition,return_latent=False):
        h=x;ld=j.zeros(x.shape[0])
        for i in range(self.layers):
            # Full-width masks avoid TPU compiler failures in the gradient of
            # unaligned roll/slice/concatenate operations (D=52 or 72).
            mask=j.asarray((np.arange(self.dim)+i)%2,dtype=h.dtype)
            t=nn.Dense(self.width,name=f'a{i}')(h*mask)
            if condition.shape[-1]:t=t+nn.Dense(self.width,name=f'c{i}')(condition)
            t=nn.silu(t);t=nn.silu(nn.Dense(self.width,name=f'b{i}')(t))
            shift=nn.Dense(self.dim,kernel_init=nn.initializers.zeros,name=f'shift{i}')(t)
            raw=nn.Dense(self.dim,kernel_init=nn.initializers.zeros,name=f'scale{i}')(t)
            scale=(1-mask)*self.scale_bound*j.tanh(raw/self.scale_bound)
            h=mask*h+(1-mask)*(h*j.exp(scale)+shift);ld=ld+scale.sum(-1)
        if return_latent:return h
        energy=j.square(h).sum(-1)
        return .5*(energy+self.dim*j.log(2*j.pi))-ld,energy,ld

def prepare(features,labels,kind,floor,transform='linear'):
    ix=splits();target={k:(v[:,20:] if kind=='power' else v) for k,v in features.items()}
    offset=np.median(target['train'],axis=0);robust_scale=np.maximum(np.subtract(*np.quantile(target['train'],[.75,.25],axis=0))/1.349,1e-8)
    values={k:np.arcsinh((v-offset)/robust_scale) if transform=='asinh' else v for k,v in target.items()}
    mean,white=whitening(values['train'],floor)
    if kind=='cosmology':conditions={k:labels[v,:2] for k,v in ix.items()}
    elif kind=='power':conditions={k:v[:,:20] for k,v in features.items()}
    else:conditions={k:np.empty((len(v),0)) for k,v in features.items()}
    cm=conditions['train'].mean(0);cs=np.maximum(conditions['train'].std(0),1e-6)
    z={k:((v-mean)@white).astype(np.float32) for k,v in values.items()};c={k:((v-cm)/cs).astype(np.float32) for k,v in conditions.items()}
    white_ld=np.linalg.slogdet(white)[1]
    jac={k:white_ld-np.sum(np.log(robust_scale)+.5*np.log1p(((v-offset)/robust_scale)**2),axis=1) if transform=='asinh' else np.full(len(v),white_ld) for k,v in target.items()}
    return z,c,jac,dict(mean=mean,whiten=white,condition_mean=cm,condition_std=cs,target_offset=offset,target_scale=robust_scale)

def predictor(run):
    run=Path(run);cfg=json.loads((run/'config.json').read_text())
    with np.load(run/'normalization.npz') as f:norm={k:f[k] for k in f.files}
    params=jax.tree.map(j.asarray,serialization.msgpack_restore((run/'best.msgpack').read_bytes()))
    net=AffineFlow(cfg['dim'],cfg['layers'],cfg['width'],cfg.get('scale_bound',1.5));apply=jax.jit(lambda x,c:net.apply({'params':params},x,c))
    kind=cfg['condition'];grid=norm['cosmology_grid']
    def predict(f):
        raw=f[:,20:] if kind=='power' else f;white_ld=np.linalg.slogdet(norm['whiten'])[1]
        if cfg.get('transform','linear')=='asinh':
            u=(raw-norm['target_offset'])/norm['target_scale'];value=np.arcsinh(u);jac=white_ld-np.sum(np.log(norm['target_scale'])+.5*np.log1p(u*u),axis=1)
        else:value=raw;jac=np.full(len(raw),white_ld)
        z=(value-norm['mean'])@norm['whiten'];result={}
        for start in range(0,len(z),32):
            x=z[start:start+32];ff=f[start:start+32]
            if kind=='cosmology':
                c=(grid-norm['condition_mean'])/norm['condition_std'];n,e,l=apply(j.asarray(np.repeat(x,len(grid),axis=0),j.float32),j.asarray(np.tile(c,(len(x),1)),j.float32));n=np.asarray(n).reshape(len(x),-1);e=np.asarray(e).reshape(len(x),-1)
                values=dict(marginal_nll=-logsumexp(-n,axis=1)+np.log(len(grid))-jac[start:start+len(x)],profile_nll=n.min(1)-jac[start:start+len(x)],minimum_base_energy=e.min(1))
            else:
                c=(ff[:,:20]-norm['condition_mean'])/norm['condition_std'] if kind=='power' else np.empty((len(x),0))
                n,e,l=apply(j.asarray(x,j.float32),j.asarray(c,j.float32));values=dict(nll=np.asarray(n)-jac[start:start+len(x)],base_energy=np.asarray(e))
            for key,v in values.items():result.setdefault(key,[]).append(v)
        return {k:np.concatenate(v) for k,v in result.items()}
    return predict

def fit(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);features=load_features(a.features);labels=load_labels(a.raw)
    z,c,jac,norm=prepare(features,labels,a.condition,a.floor,a.transform);norm['cosmology_grid']=labels.reshape(101,256,5)[:,0,:2]
    np.savez(out/'normalization.npz',**norm)
    cfg=vars(a).copy();cfg.update(dim=z['train'].shape[1],feature_hashes={k:sha(Path(a.features)/f'{k}_latent.npy') for k in features});write_json(out/'config.json',cfg)
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    net=AffineFlow(cfg['dim'],a.layers,a.width,a.scale_bound);params=net.init(jax.random.PRNGKey(a.seed),j.zeros((a.batch,cfg['dim'])),j.zeros((a.batch,c['train'].shape[1])))['params']
    tx=optax.chain(optax.clip_by_global_norm(5),optax.adamw(optax.cosine_decay_schedule(3e-4,a.steps,alpha=.05),weight_decay=1e-4));opt=tx.init(params)
    xtrain=j.asarray(z['train']);ctrain=j.asarray(c['train']);xval=j.asarray(z['val']);cval=j.asarray(c['val'])
    def loss(p,x,condition):return net.apply({'params':p},x,condition)[0].mean()
    @jax.jit
    def update(p,o,ids):
        v,g=jax.value_and_grad(loss)(p,xtrain[ids],ctrain[ids]);u,o=tx.update(g,o,p);return optax.apply_updates(p,u),o,v
    validate=jax.jit(lambda p:loss(p,xval,cval));rng=np.random.default_rng(a.seed);best=np.inf;start=time.monotonic()
    for step in range(1,a.steps+1):
        params,opt,train=update(params,opt,rng.integers(len(xtrain),size=a.batch))
        if step==1 or step%250==0 or step==a.steps:
            value=float(validate(params))-float(jac['val'].mean());record=dict(step=step,train_nll=float(train)-float(jac['train'].mean()),val_conditional_nll=value,seconds=time.monotonic()-start)
            if not np.isfinite(value):raise FloatingPointError('Nonfinite validation likelihood')
            print(json.dumps(record),flush=True)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            if value<best:best=value;(out/'best.msgpack').write_bytes(serialization.to_bytes(params));write_json(out/'best.json',record)
    write_json(out/'complete.json',dict(steps=a.steps,best_normal_val_nll=best,seconds=time.monotonic()-start))
    evaluate(a)

def evaluate(a):
    run=Path(a.out);predict=predictor(run);features=load_features(a.features)
    cal=predict(features['calibration']);audit=predict(features['audit'])
    np.savez(run/'calibration_scores.npz',**cal);np.savez(run/'audit_scores.npz',**audit)
    report=dict(kind='normal_and_synthetic_NOT_OFFICIAL_OOD',normal=score_report(cal,audit),probes={})
    if a.probes:
        for path in Path(a.probes).glob('*.npz'):
            with np.load(path) as f:
                all_scores={};metrics={}
                for key in f.files:
                    if key=='ids':continue
                    values=predict(f[key]);metrics[key]={name:low_fpr_metric(cal[name],v) for name,v in values.items()}
                    all_scores.update({key+'__'+name:v for name,v in values.items()})
            np.savez(run/(path.stem+'_scores.npz'),**all_scores);report['probes'][path.stem]=metrics
    report['checkpoint_sha256']=sha(run/'best.msgpack');report['runtime_sha256']=sha(__file__);write_json(run/'report.json',report);print(json.dumps(report),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['fit','evaluate']);p.add_argument('--raw',required=True);p.add_argument('--features',required=True);p.add_argument('--out',required=True);p.add_argument('--probes');p.add_argument('--condition',choices=['none','cosmology','power'],default='none');p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=10000);p.add_argument('--batch',type=int,default=256);p.add_argument('--layers',type=int,default=8);p.add_argument('--width',type=int,default=128);p.add_argument('--floor',type=float,default=1e-4);p.add_argument('--transform',choices=['linear','asinh'],default='asinh');p.add_argument('--scale-bound',type=float,default=.5);a=p.parse_args()
    fit(a) if a.mode=='fit' else evaluate(a)
