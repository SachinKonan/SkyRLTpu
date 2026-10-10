"""Masked field-level flow-matching pilot, inspired by arXiv:2505.00632.

This is a new implementation, not a reproduction of their architecture. Training
uses only the official normal training maps with fresh specified shape noise.
The cache uses fixed 2x2 pooling; the density is over supported pooled pixels.
"""
import argparse
import json
from pathlib import Path
import sys,time
import numpy as np
import jax
import jax.numpy as j
from flax import linen as nn,serialization
import optax
from common import splits,write_json,sha

jax.config.update('jax_default_matmul_precision','highest')
NOISE=.4/np.sqrt(2*30*4)


class Block(nn.Module):
    width:int
    @nn.compact
    def __call__(self,x,embedding):
        h=nn.Conv(self.width,(3,3))(nn.silu(nn.GroupNorm(num_groups=4)(x)))
        h=h+nn.Dense(self.width)(nn.silu(embedding))[:,None,None,:]
        h=nn.Conv(self.width,(3,3),kernel_init=nn.initializers.variance_scaling(.1,'fan_avg','uniform'))(nn.silu(nn.GroupNorm(num_groups=4)(h)))
        shortcut=x if x.shape[-1]==self.width else nn.Conv(self.width,(1,1))(x)
        return (h+shortcut)/np.sqrt(2)


class Velocity(nn.Module):
    width:int=16
    @nn.compact
    def __call__(self,x,t,coverage):
        frequencies=j.asarray([1,2,4,8,16,32],j.float32)*j.pi
        embedding=j.concatenate([t[:,None],j.sin(t[:,None]*frequencies),j.cos(t[:,None]*frequencies)],-1)
        embedding=nn.Dense(4*self.width)(nn.silu(nn.Dense(4*self.width)(embedding)))
        h=j.stack([x,j.broadcast_to(coverage,x.shape)],-1)
        h=nn.Conv(self.width,(3,3))(h);skips=[]
        for level in range(3):
            h=Block(self.width*(level+1))(h,embedding);skips.append(h)
            h=nn.Conv(self.width*(level+2),(3,3),strides=(2,2))(h)
        h=Block(self.width*4)(h,embedding)
        for skip in reversed(skips):
            h=jax.image.resize(h,(len(h),skip.shape[1],skip.shape[2],h.shape[-1]),method='nearest')
            h=j.concatenate([h,skip],-1);h=Block(skip.shape[-1])(h,embedding)
        return nn.Conv(1,(3,3),kernel_init=nn.initializers.zeros)(nn.silu(h))[...,0]


def prepare(a):
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
    from train import load_training
    maps,mask,labels,ix=load_training(a.raw);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    h,w=mask.shape;count=mask.reshape(h//2,2,w//2,2).sum((1,3));support=count>=3
    target=np.lib.format.open_memmap(out/'clean.npy',mode='w+',dtype=np.float16,shape=(len(maps),h//2,w//2))
    start=time.monotonic();sm=ss=0.;n=0;train_ids=set(ix['train'].tolist())
    for i in range(0,len(maps),16):
        ids=np.arange(i,min(i+16,len(maps)));x=np.zeros((len(ids),h,w),np.float32);x[:,mask]=maps[ids]
        y=x.reshape(len(ids),h//2,2,w//2,2).sum((2,4))/np.maximum(count,1)
        y=(y*support).astype(np.float16);target[ids]=y
        select=np.array([int(k) in train_ids for k in ids]);v=y[select][:,support].astype(float)
        sm+=v.sum();ss+=(v*v).sum();n+=v.size
    target.flush();mean=sm/n;signal_var=ss/n-mean*mean
    scale=np.sqrt(signal_var+np.mean(NOISE**2/count[support]))
    np.savez(out/'normalization.npz',mean=mean,scale=scale,count=count,support=support,noise_sigma=NOISE)
    write_json(out/'manifest.json',dict(kind='NORMAL_ONLY_POOLED_FIELD',shape=list(target.shape),mean=mean,scale=scale,signal_variance=signal_var,valid_pixels=int(support.sum()),seconds=time.monotonic()-start,source_sha256=sha(__file__)))
    print((out/'manifest.json').read_text(),flush=True)


def fit(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);cache=Path(a.cache)
    cfg=vars(a);write_json(out/'config.json',cfg);(out/'source.py').write_bytes(Path(__file__).read_bytes())
    with np.load(cache/'normalization.npz') as f:norm={k:f[k] for k in f.files}
    np.savez(out/'normalization.npz',**norm);ix=splits();clean=np.load(cache/'clean.npy',mmap_mode='r')
    support=j.asarray(norm['support']);coverage=j.asarray(norm['count']/4*norm['support'],j.float32)
    noise=j.asarray(NOISE/np.sqrt(np.maximum(norm['count'],1)),j.float32)
    m=norm['mean'];s=norm['scale'];xtrain=j.asarray(clean[ix['train']],j.float32)
    val_ids=np.random.default_rng(9183).permutation(ix['val'])[:a.val_count];xval=np.asarray(clean[val_ids],np.float32)
    net=Velocity(a.width);params=net.init(jax.random.PRNGKey(a.seed),j.zeros((a.batch,*support.shape)),j.zeros(a.batch),coverage)['params']
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(optax.warmup_cosine_decay_schedule(1e-5,2e-4,min(200,max(a.steps//10,1)),max(a.steps,3),end_value=2e-5),weight_decay=1e-4));opt=tx.init(params);ema=params
    def loss(p,clean,key):
        nk,ek,tk=jax.random.split(key,3)
        data=((clean+jax.random.normal(nk,clean.shape)*noise-m)/s)*support
        epsilon=jax.random.normal(ek,clean.shape)*support;t=jax.random.uniform(tk,(len(data),),minval=.001,maxval=.999)
        x=(1-t[:,None,None])*data+t[:,None,None]*epsilon
        pred=net.apply({'params':p},x,t,coverage)*support
        return j.sum(j.square(pred-(epsilon-data))*support)/(len(data)*support.sum())
    @jax.jit
    def update(p,e,o,all_train,ids,key):
        # The multi-GB data cache is an input buffer, never a closed-over HLO constant.
        value,g=jax.value_and_grad(loss)(p,all_train[ids],key);u,o=tx.update(g,o,p);p=optax.apply_updates(p,u)
        e=jax.tree.map(lambda old,new:.995*old+.005*new,e,p)
        return p,e,o,value
    validation=jax.jit(loss);rng=np.random.default_rng(a.seed);best=np.inf;start=time.monotonic()
    for step in range(1,a.steps+1):
        params,ema,opt,value=update(params,ema,opt,xtrain,rng.integers(len(xtrain),size=a.batch),jax.random.PRNGKey(a.seed*10000000+step))
        if step==1 or step%25==0:
            record=dict(step=step,train_fm_mse=float(value),seconds=time.monotonic()-start)
            if not np.isfinite(record['train_fm_mse']):raise FloatingPointError('Nonfinite training loss')
            print(json.dumps(record),flush=True)
        if step==1 or step%a.eval_every==0 or step==a.steps:
            values=[float(validation(ema,j.asarray(xval[i:i+a.batch]),jax.random.PRNGKey(6700+i))) for i in range(0,len(xval),a.batch)]
            v=float(np.mean(values));record=dict(step=step,val_fm_mse=v,seconds=time.monotonic()-start)
            with (out/'history.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('VALIDATION',json.dumps(record),flush=True)
            if v<best:best=v;(out/'best.msgpack').write_bytes(serialization.to_bytes(ema));write_json(out/'best.json',record)
    (out/'last-state.msgpack').write_bytes(serialization.to_bytes(dict(params=params,ema=ema,opt=opt,step=a.steps)))
    write_json(out/'complete.json',dict(steps=a.steps,best_normal_val_fm_mse=best,seconds=time.monotonic()-start))


def ode_score(velocity,x,mask,steps,probes):
    """Heun integration, data -> standard normal; trace probes fixed per map.

    probes is [P,H,W], independent of batch length/order. Missing pixels are
    excluded from the subspace, flow and prior. NLL = prior_NLL - integral div.
    """
    dt=1./steps
    def vf(x,t):return velocity(x,j.full((len(x),),t))*mask
    def terms(x,t):
        v=vf(x,t)
        traces=[]
        for e in probes:
            e=j.broadcast_to(e*mask,x.shape);_,jvp=jax.jvp(lambda z:vf(z,t),(x,),(e,))
            traces.append(j.sum(e*jvp,axis=(1,2)))
        return v,j.mean(j.stack(traces),0)
    def step(i,state):
        h,ld=state;v0,d0=terms(h,i*dt);v1,d1=terms(h+dt*v0,(i+1)*dt)
        return (h+.5*dt*(v0+v1))*mask,ld+.5*dt*(d0+d1)
    z,ld=jax.lax.fori_loop(0,steps,step,(x*mask,j.zeros(len(x))))
    energy=j.sum(z*z*mask,(1,2));nll=.5*(energy+mask.sum()*j.log(2*j.pi))-ld
    return nll,energy,ld


def check(a):
    # Analytic independent Gaussian flow: x(t)=sigma(t)*z with sigma linear.
    mask=j.asarray([[1,1,0],[1,0,1]],j.float32);sigma=1.7
    x=j.asarray(np.random.default_rng(17).normal(size=(3,2,3)),j.float32)*mask
    def velocity(v,t):return (1-sigma)/(sigma+(1-sigma)*t[:,None,None])*v
    expected=.5*j.sum((x/sigma)**2*mask,(1,2))+mask.sum()*(j.log(sigma)+.5*j.log(2*j.pi))
    probes=j.asarray(np.random.default_rng(18).choice([-1.,1.],size=(2,2,3)),j.float32)
    records=[]
    for steps in (8,16,32,64):
        fn=jax.jit(lambda x:ode_score(velocity,x,mask,steps,probes)[0]);scores=fn(x)
        records.append(dict(steps=steps,max_absolute_error=float(j.max(j.abs(scores-expected)))))
        assert np.allclose(scores[:1],fn(x[:1]),rtol=1e-5,atol=1e-5)
        assert np.allclose(scores,fn(x[::-1])[::-1],rtol=1e-5,atol=1e-5)
    assert records[-1]['max_absolute_error']<1e-3
    write_json(a.out,dict(kind='ANALYTIC_GAUSSIAN_SOLVER_CHECK',records=records,independence_verified=True));print(Path(a.out).read_text())


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('mode',choices=['prepare','fit','check']);p.add_argument('--raw');p.add_argument('--cache');p.add_argument('--out',required=True);p.add_argument('--steps',type=int,default=4000);p.add_argument('--batch',type=int,default=8);p.add_argument('--width',type=int,default=16);p.add_argument('--seed',type=int,default=0);p.add_argument('--val-count',type=int,default=128);p.add_argument('--eval-every',type=int,default=250);a=p.parse_args();{'prepare':prepare,'fit':fit,'check':check}[a.mode](a)
