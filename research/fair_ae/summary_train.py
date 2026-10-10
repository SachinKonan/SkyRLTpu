"""Learn a compact denoising representation of physical summaries, with five-label supervision."""
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import linen as nn,serialization
import optax
from train import load_training
from features import make_predict

class SummaryNet(nn.Module):
    latent: int=32
    dimensions: int=72
    @nn.compact
    def __call__(self,x):
        h=nn.gelu(nn.Dense(256,name='encoder1')(x))
        h=nn.gelu(nn.Dense(128,name='encoder2')(h))
        z=nn.Dense(self.latent,name='latent')(h)
        r=nn.Dense(self.dimensions,name='reconstruction')(nn.gelu(nn.Dense(128,name='decoder')(z)))
        y=nn.Dense(5,name='physics')(nn.gelu(nn.Dense(64,name='physics_hidden')(z)))
        return z,r,y


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--features',required=True);p.add_argument('--out',required=True);p.add_argument('--physics-weight',type=float,default=.25);p.add_argument('--steps',type=int,default=10000);a=p.parse_args()
    root=Path(a.features);out=Path(a.out);out.mkdir(exist_ok=False)
    cfg=dict(architecture='summary',latent=32,dimensions=72,steps=a.steps,physics_weight=a.physics_weight,seed=0,source_features=str(root))
    (out/'config.json').write_text(json.dumps(cfg,indent=2));(out/'source.py').write_bytes(Path(__file__).read_bytes())
    maps,mask,labels,splits=load_training(a.raw);extract=make_predict(mask,16)
    targets={};inputs={}
    for split in ('train','val'):
        inputs[split]=np.load(root/(split+'_latent.npy'))
        cache=root/(split+'_clean_features.npy')
        if not cache.exists():
            values=[];ids=splits[split]
            for i in range(0,len(ids),16):
                ix=ids[i:i+16];clean=np.zeros((len(ix),*mask.shape),np.float32);clean[:,mask]=maps[ix]
                values.append(extract(clean))
            np.save(cache,np.concatenate(values))
        targets[split]=np.load(cache)
    xm=inputs['train'].mean(0);xs=np.maximum(inputs['train'].std(0),1e-6)
    tm=targets['train'].mean(0);ts=np.maximum(targets['train'].std(0),1e-6)
    ym=labels[splits['train']].mean(0);ys=np.maximum(labels[splits['train']].std(0),1e-6)
    norm=dict(input_center=xm.tolist(),input_scale=xs.tolist(),target_center=tm.tolist(),target_scale=ts.tolist(),label_mean=ym.tolist(),label_std=ys.tolist())
    (out/'normalization.json').write_text(json.dumps(norm,indent=2))
    x={k:j.asarray((v-xm)/xs,j.float32) for k,v in inputs.items()};t={k:j.asarray((v-tm)/ts,j.float32) for k,v in targets.items()};y={k:j.asarray((labels[splits[k]]-ym)/ys,j.float32) for k in x}
    net=SummaryNet();params=net.init(jax.random.PRNGKey(0),j.zeros((128,72)))['params']
    tx=optax.chain(optax.clip_by_global_norm(2),optax.adamw(optax.cosine_decay_schedule(1e-3,a.steps,alpha=.05),weight_decay=1e-4));opt=tx.init(params)
    def loss(p,x,t,y):
        z,r,pred=net.apply({'params':p},x)
        rec=j.mean((r-t)**2);phy=j.mean((pred-y)**2,axis=0)
        return rec+a.physics_weight*phy.mean(),j.concatenate([j.array([rec]),phy])
    @jax.jit
    def update(p,o,ids,key):
        xb=x['train'][ids]
        # Summary corruption is a training regularizer, not a simulated OOD label.
        k1,k2=jax.random.split(key)
        xb=xb*jax.random.bernoulli(k1,.95,xb.shape)/.95+jax.random.normal(k2,xb.shape)*.02
        (v,m),g=jax.value_and_grad(loss,has_aux=True)(p,xb,t['train'][ids],y['train'][ids]);u,o=tx.update(g,o,p)
        return optax.apply_updates(p,u),o,v,m
    val=jax.jit(lambda p:loss(p,x['val'],t['val'],y['val']))
    rng=np.random.default_rng(0);best=np.inf;start=time.monotonic()
    for step in range(1,a.steps+1):
        ids=rng.integers(len(x['train']),size=128);params,opt,v,m=update(params,opt,ids,jax.random.PRNGKey(step))
        if step%500==0 or step==1 or step==a.steps:
            vl,ms=val(params);vl=float(vl);ms=np.asarray(ms)
            if not np.isfinite(vl):raise FloatingPointError('Validation became nonfinite')
            rec=dict(step=step,val_objective=vl,clean_summary_nmse=float(ms[0]),parameter_normalized_mse=ms[1:].tolist(),parameter_rmse=(np.sqrt(ms[1:])*ys).tolist(),seconds=time.monotonic()-start)
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(rec)+'\n')
            print(json.dumps(rec),flush=True)
            if vl<best:
                best=vl;(out/'best.msgpack').write_bytes(serialization.to_bytes(params));(out/'best.json').write_text(json.dumps(rec,indent=2))
    (out/'complete.json').write_text(json.dumps(dict(steps=a.steps,best=best)))

if __name__=='__main__':main()
