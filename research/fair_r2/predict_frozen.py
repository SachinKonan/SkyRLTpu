"""Run packaged candidate 04 or 05 on independent noisy maps; never fit.

Expected bundle layout is described by the adjacent REPRODUCE.json. The CLI
accepts [N,132019] or [N,1424,176] NPY arrays. Input order is preserved.
"""
import argparse
import hashlib
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import jax
from common import sha, write_json


def main(a):
    root = Path(a.bundle)
    out = Path(a.out)
    if out.exists():
        raise FileExistsError(out)
    if a.variant == '04':
        from detector import FrozenFlowDetector
        model = FrozenFlowDetector(root/'models',root/'mask.npy',root/'summary-manifest.json')
    else:
        from hybrid_detector import FrozenHybridDetector
        model = FrozenHybridDetector(root/'models',root/'mask.npy',root/'summary-manifest.json',
                                     root/'hybrid-manifest.json',root/'nll-normal-regressors.npz')
    data = np.load(a.data,mmap_mode='r',allow_pickle=False)
    if not ((data.ndim == 2 and data.shape[1:] == (int(model.mask.sum()),)) or
            (data.ndim == 3 and data.shape[1:] == model.mask.shape)):
        raise ValueError('Input must be a batch of valid-pixel vectors or full maps')
    scores = np.empty(len(data),np.float64)
    start = time.monotonic()
    for lo in range(0,len(data),a.batch):
        scores[lo:lo+a.batch] = model.predict(data[lo:lo+a.batch])
        if lo % 1024 == 0:
            print(json.dumps(dict(completed=min(lo+a.batch,len(data)),total=len(data),
                                  seconds=time.monotonic()-start)),flush=True)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite predictions')
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'x',zipfile.ZIP_DEFLATED) as z:
        z.writestr('result.json',json.dumps(dict(ood_scores=scores.tolist()),allow_nan=False))
    digest = hashlib.sha256()
    with Path(a.data).open('rb') as f:
        for block in iter(lambda:f.read(2**24),b''):
            digest.update(block)
    write_json(str(out)+'.receipt.json',dict(count=len(scores),variant=a.variant,
               score_archive_sha256=sha(out),input_sha256=digest.hexdigest(),
               summary_manifest_sha256=sha(root/'summary-manifest.json'),
               hybrid_manifest_sha256=sha(root/'hybrid-manifest.json') if a.variant=='05' else None,
               backend=jax.default_backend(),jax_version=jax.__version__,
               seconds=time.monotonic()-start,model_or_test_fitting=False,submitted=False))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--variant',choices=['04','05'],required=True)
    for name in ('bundle','data','out'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--batch',type=int,default=16)
    args=p.parse_args()
    if args.batch < 1:
        p.error('--batch must be positive')
    main(args)
