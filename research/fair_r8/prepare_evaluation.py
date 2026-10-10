"""Assemble exact existing R2/R4 observations; no public data, new noise or fit."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np


def sha(path):
    d=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(16*1024*1024),b''):d.update(b)
    return d.hexdigest()


def main(a):
    r2,r4,out=map(Path,(a.benchmark,a.closure,a.out));out.mkdir(parents=True,exist_ok=False)
    old=np.load(r2/'observed.npy',mmap_mode='r');closure=np.load(r4/'observed.npy',mmap_mode='r')
    om=json.loads((r2/'manifest.json').read_text());cm=json.loads((r4/'manifest.json').read_text())
    oi=np.load(r2/'source_ids.npy');ci=np.load(r4/'source_ids.npy')
    selections=[('fit_normal',old,oi,om['groups']['development/normal']),
                ('calibration',old,oi,om['groups']['calibration'])]
    selections.extend((name,closure,ci,d) for name,d in cm['groups'].items())
    assert sum(d['count'] for *_,d in selections)==13440
    maps=np.lib.format.open_memmap(out/'observed.npy',mode='w+',dtype=np.float32,shape=(13440,*old.shape[1:]))
    ids=np.empty(13440,np.int32);groups={};offset=0
    for name,source,sids,d in selections:
        n=d['count'];lo=d['offset'];groups[name]=dict(offset=offset,count=n)
        for i in range(0,n,32):
            hi=min(i+32,n);maps[offset+i:offset+hi]=source[lo+i:lo+hi]
            assert np.array_equal(maps[offset+i:offset+hi],source[lo+i:lo+hi])
        ids[offset:offset+n]=sids[lo:lo+n];offset+=n
    maps.flush();np.save(out/'source_ids.npy',ids)
    report=dict(kind='R8_EXISTING_NORMAL_AND_REUSED_SYNTHETIC_DIAGNOSTICS',public_test_read=False,
                exact_copy_verified=True,shape=list(maps.shape),groups=groups,
                observations_sha256=sha(out/'observed.npy'),source_ids_sha256=sha(out/'source_ids.npy'),
                source_manifests={str(r):sha(r/'manifest.json') for r in (r2,r4)},source_sha256=sha(__file__))
    (out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('benchmark','closure','out'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
