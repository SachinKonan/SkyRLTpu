"""Resolve the 64-vs-128-step question raised by the normal-data pilot."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from field_latents import FrozenNoiseField
from noise_statistics import statistics
from field_flow import NOISE
from common import write_json,sha


def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    with np.load(Path(a.pilot)/'statistics-64.npz',allow_pickle=False) as f:
        previous={k:f[k] for k in f.files};ids=previous.pop('ids')
    model=FrozenNoiseField(a.run,steps=128);n=model.norm
    clean=np.load(Path(a.cache)/'clean.npy',mmap_mode='r')[ids].astype(np.float32)
    x=(clean+np.random.default_rng(96831).normal(size=clean.shape)*NOISE/
       np.sqrt(np.maximum(n['count'],1)))*model.support
    start=time.monotonic();values=statistics(model(x.astype(np.float32)),model.support)
    elapsed=time.monotonic()-start
    np.savez(out/'statistics-128.npz',ids=ids,**values)
    error={k:dict(median_absolute=float(np.median(abs(v-previous[k]))),
                  maximum_absolute=float(np.max(abs(v-previous[k]))),
                  median_over_normal_sd=float(np.median(abs(v-previous[k]))/max(v.std(),1e-8)))
           for k,v in values.items()}
    report=dict(kind='R6_NORMAL_ONLY_NUMERICAL_REFINEMENT',public_test_read=False,
                count=len(ids),seconds=elapsed,checkpoint_sha256=sha(Path(a.run)/'best.msgpack'),
                source_sha256=sha(__file__),difference64_128=error)
    write_json(out/'report.json',report);print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('run','cache','pilot','out'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
