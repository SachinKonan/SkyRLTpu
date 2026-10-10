"""Train-only supervised denoising AE. Never opens the official test file."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as jnp
from flax import linen as nn, serialization
import optax

jax.config.update("jax_default_matmul_precision", "highest")

NOISE = .4 / np.sqrt(2 * 30 * 2**2)

class Autoencoder(nn.Module):
    latent_dim: int = 64
    width: int = 16

    @nn.compact
    def __call__(self, x, mask):
        # Fixed geometry; the mask is a channel, never an inferred test statistic.
        x = jnp.pad(x, ((0, 0), (0, 16), (0, 16)))
        m = jnp.pad(mask, ((0, 16), (0, 16)))
        h = jnp.stack([x, jnp.broadcast_to(m, x.shape)], axis=-1)
        channels = [self.width * a for a in (1, 2, 3, 4, 6)]
        for i, c in enumerate(channels):
            h = nn.Conv(c, (5, 5) if i == 0 else (3, 3), strides=(2, 2), name=f'enc{i}')(h)
            h = nn.gelu(nn.GroupNorm(num_groups=4, name=f'enorm{i}')(h))
        z = nn.Dense(self.latent_dim, name='latent')(h.reshape((h.shape[0], -1)))
        theta = nn.Dense(64, name='physics_hidden')(z)
        theta = nn.Dense(2, name='physics_out')(nn.gelu(theta))
        h = nn.Dense(45 * 6 * channels[-1], name='decode_dense')(z)
        h = h.reshape((-1, 45, 6, channels[-1]))
        for i, c in enumerate(channels[::-1]):
            h = jax.image.resize(h, (h.shape[0], h.shape[1]*2, h.shape[2]*2, h.shape[3]), method='nearest')
            h = nn.Conv(c, (3, 3), name=f'dec{i}')(h)
            h = nn.gelu(nn.GroupNorm(num_groups=4, name=f'dnorm{i}')(h))
        recon = nn.Conv(1, (3, 3), name='output')(h)[:, :1424, :176, 0]
        return z, recon, theta


def load_training(raw):
    raw = Path(raw)
    files = {'WIDE12H_bin2_2arcmin_kappa_newrealization.npy': '330ccc9ced8288b1d4d9bd81e74228b0',
             'WIDE12H_bin2_2arcmin_mask.npy': '29b8bbb8a9ad41d3dacfaf61008a8268',
             'label_newrealization.npy': '2b88e3ccf8c8e66b445ee649886586c7'}
    for name, expected in files.items():
        h = hashlib.md5()
        with (raw/name).open('rb') as f:
            for block in iter(lambda: f.read(2**24), b''): h.update(block)
        if h.hexdigest() != expected: raise ValueError(f'Checksum mismatch: {name}')
    maps = np.load(raw / next(iter(files)), mmap_mode='r', allow_pickle=False)
    mask = np.load(raw/'WIDE12H_bin2_2arcmin_mask.npy', allow_pickle=False)
    labels = np.load(raw/'label_newrealization.npy', allow_pickle=False)
    assert maps.shape == (101,256,int(mask.sum())) and mask.shape == (1424,176)
    order = np.random.default_rng(2026).permutation(256)
    splits = {}
    for name, ids in [('train',order[:160]),('val',order[160:192]),('calibration',order[192:224]),('audit',order[224:])]:
        splits[name] = np.array([c*256+r for c in range(101) for r in ids])
    return maps.reshape((-1,maps.shape[-1])), mask, labels.reshape((-1,5)), splits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--steps', type=int, default=6000); ap.add_argument('--batch', type=int, default=16)
    ap.add_argument('--latent',type=int,default=64); ap.add_argument('--width',type=int,default=16)
    ap.add_argument('--physics-weight',type=float,default=.25); ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--eval-every',type=int,default=250); ap.add_argument('--val-count',type=int,default=1024)
    a=ap.parse_args(); out=Path(a.out); out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(vars(a),indent=2))
    (out/'source.py').write_bytes(Path(__file__).read_bytes())
    print('devices',jax.devices(),flush=True)
    maps,mask,labels,splits=load_training(a.raw)
    # Exact train-only global moments, streamed to avoid large temporary arrays.
    sm=ss=0.; n=0
    for idx in np.array_split(splits['train'],128):
        x=np.asarray(maps[idx],np.float64); sm+=x.sum(); ss+=np.square(x).sum(); n+=x.size
    mean=sm/n; scale=np.sqrt(ss/n-mean**2); input_scale=np.sqrt(scale**2+NOISE**2)
    ym=labels[splits['train'],:2].mean(0); ys=labels[splits['train'],:2].std(0)
    norm=dict(mean=float(mean),signal_scale=float(scale),input_scale=float(input_scale),label_mean=ym.tolist(),label_std=ys.tolist(),noise_sigma=float(NOISE))
    (out/'normalization.json').write_text(json.dumps(norm,indent=2))
    np.savez(out/'splits.npz',**splits)
    net=Autoencoder(a.latent,a.width)
    key=jax.random.PRNGKey(a.seed); key,init=jax.random.split(key)
    m=jnp.asarray(mask,dtype=jnp.float32)
    params=net.init(init,jnp.zeros((a.batch,1424,176)),m)['params']
    lr=optax.warmup_cosine_decay_schedule(1e-5,3e-4,min(200,max(1,a.steps//10)),max(3,a.steps),end_value=3e-5)
    tx=optax.chain(optax.clip_by_global_norm(1.),optax.adamw(lr,weight_decay=1e-4))
    opt=tx.init(params)
    def objective(p,clean,y,key):
        noise=jax.random.normal(key,clean.shape)*NOISE
        x=((clean+noise-mean)/input_scale)*m
        _,rec,pred=net.apply({'params':p},x,m)
        target=((clean-mean)/scale)*m
        rec_loss=jnp.sum(jnp.square(rec-target)*m)/(len(clean)*m.sum())
        per_parameter=jnp.mean(jnp.square(pred-y),axis=0)
        loss=rec_loss+a.physics_weight*jnp.mean(per_parameter)
        return loss,jnp.concatenate((jnp.array([rec_loss]),per_parameter))
    @jax.jit
    def update(p,o,clean,y,key):
        (loss,metrics),g=jax.value_and_grad(objective,has_aux=True)(p,clean,y,key)
        updates,o=tx.update(g,o,p)
        return optax.apply_updates(p,updates),o,loss,metrics
    evaluate=jax.jit(objective)
    rng=np.random.default_rng(a.seed)
    val_ids=np.random.default_rng(871).permutation(splits['val'])[:a.val_count]
    def batch(ids):
        clean=np.zeros((len(ids),1424,176),np.float32); clean[:,mask]=maps[ids]
        return clean,((labels[ids,:2]-ym)/ys).astype(np.float32)
    best=float('inf'); start=time.monotonic()
    for step in range(1,a.steps+1):
        ids=rng.choice(splits['train'],a.batch,replace=False)
        clean,y=batch(ids); key,sub=jax.random.split(key)
        params,opt,loss,metrics=update(params,opt,clean,y,sub)
        if step==1 or step%25==0:
            value=float(loss)
            if not np.isfinite(value): raise FloatingPointError('Nonfinite training loss')
            print(json.dumps(dict(step=step,train_loss=value,metrics=np.asarray(metrics).tolist(),seconds=time.monotonic()-start)),flush=True)
        if step==1 or step%a.eval_every==0 or step==a.steps:
            totals=np.zeros(4); count=0
            for k in range(0,len(val_ids)-a.batch+1,a.batch):
                clean,y=batch(val_ids[k:k+a.batch]); v,ms=evaluate(params,clean,y,jax.random.PRNGKey(90210+k))
                totals+=np.r_[float(v),np.asarray(ms)]; count+=1
            if not count: raise ValueError('Validation count smaller than batch')
            vals=totals/count
            record=dict(step=step,val_objective=float(vals[0]),reconstruction_nmse=float(vals[1]),parameter_normalized_mse=vals[2:].tolist(),parameter_rmse=(np.sqrt(vals[2:])*ys).tolist(),seconds=time.monotonic()-start)
            with (out/'validation.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
            print('VALIDATION',json.dumps(record),flush=True)
            if vals[0]<best:
                best=vals[0]
                (out/'best.msgpack').write_bytes(serialization.to_bytes(params))
                (out/'best.json').write_text(json.dumps(record,indent=2))
    (out/'complete.json').write_text(json.dumps(dict(steps=a.steps,best=float(best),seconds=time.monotonic()-start)))

if __name__=='__main__':main()
