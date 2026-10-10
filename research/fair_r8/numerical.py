"""Normal-only likelihood solver and pointwise checks for a frozen R8 model."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np
import jax
sys.path.insert(1,str(Path(__file__).resolve().parents[1]/'fair_r2'))
from field_evaluate import predictor
from field_flow import NOISE
from common import splits,sha,write_json


def main(a):
    run,cache,out=map(Path,(a.run,a.cache,a.out));out.mkdir(parents=True,exist_ok=False)
    with np.load(run/'normalization.npz') as f:norm={k:f[k] for k in f.files}
    ids=np.random.default_rng(862).permutation(splits()['val'])[:32]
    clean=np.load(cache/'clean.npy',mmap_mode='r')
    observed=(clean[ids]+np.random.default_rng(18962).normal(size=(len(ids),*norm['support'].shape))*
              NOISE/np.sqrt(np.maximum(norm['count'],1)))*norm['support']
    observations=np.asarray(observed,np.float32);records={};values={}
    for steps in (32,64,128):
        start=time.monotonic();fn=predictor(run,steps,batch=16);v=fn(observations)
        for k,x in v.items():assert np.isfinite(x).all()
        compile_first=time.monotonic()-start;start=time.monotonic();repeat=fn(observations);seconds=time.monotonic()-start
        assert all(np.array_equal(v[k],repeat[k]) for k in v)
        records[str(steps)]=dict(compile_and_first_seconds=compile_first,warm_seconds=seconds,
                                nll_mean=float(v['nll'].mean()),nll_std=float(v['nll'].std()))
        if steps==64:
            alone=fn(observations[:1]);reverse=fn(observations[::-1]);a1=fn(observations[:5]);a2=fn(observations[5:])
            errors={k:dict(singleton=float(np.max(abs(v[k][:1]-alone[k]))),
                           reverse=float(np.max(abs(v[k]-reverse[k][::-1]))),
                           chunks=float(np.max(abs(v[k]-np.r_[a1[k],a2[k]])))) for k in v}
            assert max(vv for check in errors.values() for vv in check.values())<.05,errors
            records[str(steps)]['pointwise_max_absolute_errors']=errors
        values[steps]=v;np.savez(out/f'steps{steps}.npz',ids=ids,**v)
        print(json.dumps(dict(stage='numerical',steps=steps,**records[str(steps)])),flush=True)
        del fn;jax.clear_caches()
    for steps in (32,64):
        delta=values[steps]['nll']-values[128]['nll'];sd=values[128]['nll'].std()
        records[str(steps)]['nll_difference_vs128']=dict(median_absolute=float(np.median(abs(delta))),
            max_absolute=float(np.max(abs(delta))),median_over_normal_sd=float(np.median(abs(delta))/sd),
            centered_rms_over_normal_sd=float(np.std(delta)/sd))
    # Conservative fixed criterion for proceeding with a 64-step likelihood.
    gate=records['64']['nll_difference_vs128']['median_over_normal_sd']<.003
    report=dict(kind='R8_NORMAL_ONLY_NUMERICAL_CHECK',public_test_read=False,synthetic_anomalies_read=False,
                count=len(ids),checkpoint_sha256=sha(run/'best.msgpack'),source_sha256=sha(__file__),
                records=records,proceed64=gate,
                gate='Median absolute 64-to-128 NLL difference below 0.3 percent of normal-pilot NLL SD. This finite pilot does not prove exact integration.')
    write_json(out/'report.json',report)
    assert gate,report


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('run','cache','out'):p.add_argument('--'+n,required=True)
    main(p.parse_args())
