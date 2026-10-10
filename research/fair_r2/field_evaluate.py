"""Check field likelihood cost/convergence using normal validation maps only."""
import argparse,json,time
from pathlib import Path
import numpy as np
import jax
import jax.numpy as j
from flax import serialization
from field_flow import Velocity,ode_score,NOISE
from common import splits,write_json,sha


def predictor(run,steps=32,probe_seed=451,batch=8):
    run=Path(run);cfg=json.loads((run/'config.json').read_text())
    with np.load(run/'normalization.npz') as f:norm={k:f[k] for k in f.files}
    params=jax.tree.map(j.asarray,serialization.msgpack_restore((run/'best.msgpack').read_bytes()))
    mask=j.asarray(norm['support'],j.float32);coverage=j.asarray(norm['count']/4*norm['support'],j.float32)
    net=Velocity(cfg['width']);velocity=lambda x,t:net.apply({'params':params},x,t,coverage)
    probes=j.asarray(np.random.default_rng(probe_seed).choice([-1.,1.],size=(1,*mask.shape)),j.float32)
    apply=jax.jit(lambda x:ode_score(velocity,x,mask,steps,probes))
    def predict(x):
        values=[]
        for start in range(0,len(x),batch):
            q=x[start:start+batch];n=len(q)
            if n<batch:q=np.pad(q,((0,batch-n),(0,0),(0,0)))
            q=(q-norm['mean'])/norm['scale']*norm['support']
            values.append(np.column_stack([np.asarray(v)[:n] for v in apply(j.asarray(q,j.float32))]))
        values=np.concatenate(values)
        return dict(nll=values[:,0]+int(norm['support'].sum())*np.log(float(norm['scale'])),base_energy=values[:,1],integrated_divergence=values[:,2])
    return predict


def main(a):
    run=Path(a.run);out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    with np.load(run/'normalization.npz') as f:norm={k:f[k] for k in f.files}
    cache=np.load(Path(a.cache)/'clean.npy',mmap_mode='r');ids=np.random.default_rng(862).permutation(splits()['val'])[:a.count]
    clean=np.asarray(cache[ids],np.float32)
    observed=(clean+np.random.default_rng(18962).normal(size=clean.shape)*NOISE/np.sqrt(np.maximum(norm['count'],1)))*norm['support']
    results={};reports=[]
    for steps in (16,32,64):
        start=time.monotonic();fn=predictor(run,steps,batch=a.batch);v=fn(observed);first=time.monotonic()-start
        start=time.monotonic();repeat=fn(observed);seconds=time.monotonic()-start
        assert np.isfinite(np.column_stack(list(v.values()))).all()
        assert all(np.allclose(v[k],repeat[k]) for k in v)
        if steps==32:
            alone=fn(observed[:1]);reverse=fn(observed[::-1]);chunks={k:np.concatenate([fn(observed[:3])[k],fn(observed[3:])[k]]) for k in v}
            for k in v:
                assert np.allclose(v[k][:1],alone[k],rtol=1e-5,atol=1e-3)
                assert np.allclose(v[k],reverse[k][::-1],rtol=1e-5,atol=1e-3)
                assert np.allclose(v[k],chunks[k],rtol=1e-5,atol=1e-3)
        np.savez(out/f'steps{steps}.npz',ids=ids,**v);results[steps]=v
        record=dict(steps=steps,count=len(observed),compile_and_first_seconds=first,warm_seconds=seconds,maps_per_second=len(observed)/seconds,nll_mean=float(v['nll'].mean()),nll_std=float(v['nll'].std()))
        print(json.dumps(record),flush=True);reports.append(record);del fn;jax.clear_caches()
    for record in reports[:-1]:
        v=results[record['steps']]['nll'];ref=results[64]['nll'];record.update(median_abs_nll_error_vs64=float(np.median(abs(v-ref))),max_abs_nll_error_vs64=float(np.max(abs(v-ref))),normal_std_fraction=float(np.median(abs(v-ref))/max(ref.std(),1e-8)))
    write_json(out/'report.json',dict(kind='NORMAL_VALIDATION_NUMERICAL_CHECK_NOT_OOD_SCORE',checkpoint_sha256=sha(run/'best.msgpack'),independence_verified=True,trace_probes=1,trace_seed=451,records=reports))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--cache',required=True);p.add_argument('--out',required=True);p.add_argument('--count',type=int,default=16);p.add_argument('--batch',type=int,default=8);main(p.parse_args())
