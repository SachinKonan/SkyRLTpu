"""Normal-only numerical audit before evaluating an alternate summary score."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
import jax.numpy as j
from continuous_profile import Profile
from flow import predictor
from common import load_labels, splits, write_json, sha


def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    ix=splits()['val'];rows=np.arange(len(ix)).reshape(101,32)[:,:2].ravel()
    f=np.load(Path(a.features)/'val_latent.npy').astype(np.float64)[rows]
    labels=load_labels(a.raw)[ix[rows],:2]
    model=Profile(a.run,steps=128);start=time.monotonic();values=model(f,labels)
    elapsed=time.monotonic()-start
    reference=predictor(a.run)(f)['minimum_base_energy']
    assert np.allclose(values['grid_energy'],reference,rtol=1e-6,atol=1e-4)
    assert np.all(values['continuous_energy']<=values['grid_energy']+1e-4)
    violation=values['continuous_theta']@model.equations[:,:2].T+model.equations[:,2]
    assert violation.max()<3e-6
    assert all(np.isfinite(v).all() for v in values.values())
    single=model(f[:1]);reverse=model(f[::-1]);pieces=[model(f[:3]),model(f[3:17])]
    checks={}
    for key in ('grid_energy','continuous_energy','continuous_theta'):
        checks[key]={'single':float(np.max(abs(values[key][:1]-single[key]))),
                     'reverse':float(np.max(abs(values[key]-reverse[key][::-1]))),
                     'chunks':float(np.max(abs(values[key][:17]-np.concatenate([v[key] for v in pieces]))))}
        assert all(v<1e-4 for v in checks[key].values()),checks
    # Projection reference checks: fixed polygon vertices project to themselves,
    # and the same CPU geometry independently projects random points.
    rng=np.random.default_rng(4741);c=rng.uniform(model.vertices.min(0)-3,model.vertices.max(0)+3,size=(200,2))
    expected=[]
    for point in c:
        if np.all(point@model.equations[:,:2].T+model.equations[:,2]<=1e-6):expected.append(point);continue
        edges=[]
        for va,vb in zip(model.vertices,np.roll(model.vertices,-1,axis=0)):
            edge=vb-va;t=np.clip(np.dot(point-va,edge)/np.dot(edge,edge),0,1);edges.append(va+t*edge)
        expected.append(min(edges,key=lambda p:np.sum((point-p)**2)))
    projected=np.asarray(model.project(j.asarray(c,j.float32)))
    assert np.allclose(projected,expected,rtol=1e-5,atol=2e-6)
    chosen=np.random.default_rng(9051).choice(len(f),16,replace=False)
    dense,n=model.dense_reference(f[chosen],resolution=51)
    short=Profile(a.run,steps=64);short_values=short(f)
    gap=values['grid_energy']-values['continuous_energy']
    np.savez(out/'scores.npz',ids=ix[rows],true_theta=labels,dense_rows=chosen,dense_energy=dense,
             short_energy=short_values['continuous_energy'],**values)
    report=dict(kind='NORMAL_ONLY_CONTINUOUS_PROFILING_PILOT',count=len(f),
        checkpoint_sha256=sha(Path(a.run)/'best.msgpack'),public_test_read=False,model_training=False,
        seconds=elapsed,max_hull_violation=float(violation.max()),pointwise_errors=checks,
        energy_reduction_quantiles=np.quantile(gap,[0,.5,.9,1]).tolist(),
        dense_reference_count=n,continuous_minus_dense_quantiles=np.quantile(values['continuous_energy'][chosen]-dense,[0,.5,.9,1]).tolist(),
        short_minus_long_quantiles=np.quantile(short_values['continuous_energy']-values['continuous_energy'],[0,.5,.9,1]).tolist(),
        geometry_projection_verified=True,
        interpretation='Known true cosmology is only an audit diagnostic; the deployed score cannot use it. Approximate continuous minimization is constrained to the training convex hull.')
    write_json(out/'report.json',report);print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('run','features','raw','out'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
