"""Evaluate one disjoint shard of frozen per-map field-noise statistics."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from field_latents import FrozenNoiseField
from noise_statistics import statistics
from common import write_json,sha


def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    normals,closure=map(Path,(a.normals,a.closure))
    nx=np.load(normals/'observed.npy',mmap_mode='r');cx=np.load(closure/'observed.npy',mmap_mode='r')
    nm=json.loads((normals/'manifest.json').read_text());cm=json.loads((closure/'manifest.json').read_text())
    assert len(nx)==6464 and len(cx)==9696
    ids=np.arange(a.shard,len(nx)+len(cx),a.shards)
    model=FrozenNoiseField(a.run,steps=a.steps,batch=8);start=time.monotonic();values={}
    for lo in range(0,len(ids),8):
        row=ids[lo:lo+8];x=np.empty((len(row),*model.support.shape),np.float32)
        normal=row<len(nx);x[normal]=nx[row[normal]];x[~normal]=cx[row[~normal]-len(nx)]
        result=statistics(model(x),model.support)
        for k,v in result.items():values.setdefault(k,[]).append(v)
        if lo%256==0:print(json.dumps(dict(completed=lo+len(row),total=len(ids),seconds=time.monotonic()-start)),flush=True)
    values={k:np.concatenate(v) for k,v in values.items()}
    assert all(v.shape==ids.shape and np.isfinite(v).all() for v in values.values())
    np.savez(out/'statistics.npz',ids=ids,**values)
    report=dict(kind='R6_FROZEN_FIELD_STATISTICS_REUSED_NORMAL_AND_SYNTHETIC',public_test_read=False,
                shard=a.shard,shards=a.shards,steps=a.steps,count=len(ids),seconds=time.monotonic()-start,
                checkpoint_sha256=sha(Path(a.run)/'best.msgpack'),
                normalization_sha256=sha(Path(a.run)/'normalization.npz'),source_sha256=sha(__file__),
                statistics_source_sha256=sha(Path(__file__).with_name('noise_statistics.py')),
                solver_source_sha256=sha(Path(__file__).with_name('field_latents.py')),
                normal_cache_sha256=nm['observations_sha256'],closure_cache_sha256=cm['observations_sha256'],
                output_sha256=sha(out/'statistics.npz'))
    write_json(out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('run','normals','closure','out'):p.add_argument('--'+name,required=True)
    p.add_argument('--shard',type=int,required=True);p.add_argument('--shards',type=int,default=8)
    p.add_argument('--steps',type=int,default=64)
    main(p.parse_args())
