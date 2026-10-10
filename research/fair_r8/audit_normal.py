"""Paired normal-only quality audit for old and newly trained field flows.

Run on the handed-over lab, after all checkpoint files have been frozen.
No synthetic or public test maps are opened.
"""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import jax
import jax.numpy as j
from flax import serialization
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'fair_r2'))
from field_flow import Velocity,NOISE
from common import splits,write_json,sha


def main(a):
    root,old,cache,out=map(Path,(a.root,a.old,a.cache,a.out));out.mkdir(parents=True,exist_ok=False)
    used=np.random.default_rng(9183).permutation(splits()['val'])[:128]
    ids=np.random.default_rng(884101).permutation(np.setdiff1d(splits()['val'],used))[:1024]
    assert not np.intersect1d(ids,used).size
    clean=np.load(cache/'clean.npy',mmap_mode='r')
    with np.load(cache/'normalization.npz') as f:norm={k:f[k] for k in f.files}
    support=j.asarray(norm['support'],j.float32);coverage=j.asarray(norm['count']/4*norm['support'],j.float32)
    rng=np.random.default_rng(884102)
    observed=((np.asarray(clean[ids],np.float32)+rng.normal(size=(len(ids),*support.shape)).astype(np.float32)*
               (NOISE/np.sqrt(np.maximum(norm['count'],1)))-norm['mean'])/norm['scale'])*norm['support']
    epsilon=rng.normal(size=observed.shape).astype(np.float32)*norm['support']
    t=rng.uniform(.001,.999,size=len(ids)).astype(np.float32)
    grid=np.asarray([.05,.15,.3,.5,.7,.85,.95],np.float32)
    names={**{f'old-s{s}':old/f'field-s{s}' for s in range(2)},
           **{f'w{w}-s{s}':root/f'field-w{w}-s{s}' for w in (16,32) for s in range(2)}}
    functions={};values={};records={};start=time.monotonic()
    for name,path in names.items():
        assert (path/'complete.json').is_file(),f'Training incomplete: {path}'
        cfg=json.loads((path/'config.json').read_text());width=cfg['width']
        assert sha(path/'normalization.npz')==sha(cache/'normalization.npz')
        params=jax.tree.map(j.asarray,serialization.msgpack_restore((path/'best.msgpack').read_bytes()))
        if width not in functions:
            net=Velocity(width)
            def error(p,x,e,t,net=net):
                xt=(1-t[:,None,None])*x+t[:,None,None]*e
                v=net.apply({'params':p},xt,t,coverage)*support
                return j.sum((v-(e-x))**2*support,axis=(1,2))/support.sum()
            functions[width]=jax.jit(error)
        fn=functions[width];result=[]
        for lo in range(0,len(ids),16):
            x=j.asarray(observed[lo:lo+16]);e=j.asarray(epsilon[lo:lo+16])
            ts=[j.asarray(t[lo:lo+16])]+[j.full((len(x),),tt) for tt in grid]
            result.append(np.stack([np.asarray(fn(params,x,e,tt)) for tt in ts],axis=1))
        matrix=np.concatenate(result);assert matrix.shape==(1024,8) and np.isfinite(matrix).all()
        values[name]=matrix
        records[name]=dict(path=str(path),checkpoint_sha256=sha(path/'best.msgpack'),
                           checkpoint=json.loads((path/'best.json').read_text()),width=width,
                           uniform_t_mse=float(matrix[:,0].mean()),grid_mse=matrix[:,1:].mean(0).tolist(),
                           seconds_since_start=time.monotonic()-start)
        print(json.dumps(dict(stage='normal_audit',name=name,**records[name])),flush=True)
    np.savez(out/'paired-errors.npz',ids=ids,uniform_t=t,grid_t=grid,**values)
    rng=np.random.default_rng(884103);difference={}
    for width in (16,32):
        delta=np.mean([values[f'w{width}-s{s}'][:,0]-values[f'old-s{s}'][:,0] for s in range(2)],axis=0)
        boot=np.array([delta[rng.integers(len(delta),size=len(delta))].mean() for _ in range(1000)])
        difference[str(width)]=dict(new_minus_old_mean_mse=float(delta.mean()),
                                    paired_percentile95=np.quantile(boot,[.025,.975]).tolist(),replicates=1000)
    write_json(out/'report.json',dict(kind='R8_NORMAL_ONLY_MODEL_QUALITY_NOT_OOD_PERFORMANCE',
        data='1024 validation maps excluding the 128 maps used for checkpoint selection; fresh paired shape noise and flow noise.',
        public_test_read=False,synthetic_anomalies_read=False,records=records,paired_mean_differences=difference,
        source_sha256=sha(__file__),output_sha256=sha(out/'paired-errors.npz'),
        limitations=['Normal validation realizations have been used by other components in earlier rounds.',
                     'The IID bootstrap measures conditional paired model differences; cosmology and simulator dependence are not fully modeled.',
                     'Better flow-matching MSE is not proof of better likelihood, latent structure or OOD detection.']))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','old','cache','out'):p.add_argument('--'+n,required=True)
    main(p.parse_args())
