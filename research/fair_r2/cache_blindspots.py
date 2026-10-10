"""Additional paired shifts evaluated after candidate 05 was frozen.

These families were seen by earlier summary-model studies. They were not part
of selecting the field/summary mixture. Normal source maps are reused.
"""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'fair_ae'))
from train import load_training,NOISE
from diagnostics import shift
from common import write_json,sha


def main(a):
    root = Path(a.root)
    out = Path(a.out)
    out.mkdir(parents=True,exist_ok=False)
    maps,mask,labels,splits = load_training(a.raw)
    h,w = mask.shape
    count = mask.reshape(h//2,2,w//2,2).sum((1,3))
    support = count >= 3
    plans = [('development',474,'val','development',
              [('gaussian_mixture',.2),('gaussian_mixture',.5),('spectral_low',.5),
               ('spectral_high',.5),('quadratic_peak',.5)]),
             ('reused',475,'audit','r1_novel_now_reused',
              [('spatial_modulation',.2),('spatial_modulation',.5),('anisotropic',.5)])]
    groups = {}
    for phase,seed,source,cache,families in plans:
        for family,strength in families:
            groups[f'{phase}/{family}:{strength}'] = dict(offset=len(groups)*512,count=512)
    pooled = np.lib.format.open_memmap(out/'observed.npy',mode='w+',dtype=np.float32,
                                      shape=(len(groups)*512,h//2,w//2))
    features = np.empty((len(groups)*512,72),np.float32)
    source_ids = np.empty(len(features),np.int32)
    start = time.monotonic()
    for phase,seed,source,cache,families in plans:
        ids = np.random.default_rng(seed).permutation(splits[source])[:512]
        with np.load(root/'probes72'/f'{cache}.npz') as f:
            assert np.array_equal(ids,f['ids'])
            for family,strength in families:
                key=f'{family}:{strength}';lo=groups[phase+'/'+key]['offset']
                features[lo:lo+512] = f[key]
                source_ids[lo:lo+512] = ids
        for i in range(0,len(ids),16):
            ix = ids[i:i+16]
            x = np.zeros((len(ix),h,w),np.float32);x[:,mask]=maps[ix]
            ri = np.array([splits['train'][160*(n//256)+(n%160)] for n in ix])
            reference = np.zeros_like(x);reference[:,mask]=maps[ri]
            rng = np.random.default_rng(7300+i+int(source=='audit'))
            noise = rng.normal(0,NOISE,x.shape).astype(np.float32)*mask
            for family,strength in families:
                y = shift(x,mask,family,strength,reference)+noise
                lo = groups[f'{phase}/{family}:{strength}']['offset']+i
                pooled[lo:lo+len(ix)] = (y.reshape(len(ix),h//2,2,w//2,2).sum((2,4))
                                          / np.maximum(count,1)*support).astype(np.float32)
        print('prepared',phase,time.monotonic()-start,flush=True)
    pooled.flush()
    np.save(out/'features.npy',features);np.save(out/'source_ids.npy',source_ids)
    write_json(out/'manifest.json',dict(kind='ADDITIONAL_PAIRED_SYNTHETIC_NOT_OFFICIAL_OOD',
        shape=list(pooled.shape),groups=groups,seconds=time.monotonic()-start,
        frozen_hybrid_manifest_sha256=sha(root/'hybrid-frozen/manifest.json'),
        source_sha256=sha(__file__),raw_test_read=False,
        limitation='Summary models and normal maps are reused; these are additional family checks for the already frozen candidate 05.'))


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    for name in ('root','raw','out'):
        p.add_argument('--'+name,required=True)
    main(p.parse_args())
