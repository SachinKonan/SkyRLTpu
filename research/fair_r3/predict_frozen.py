"""Standalone R3 inference CLI. Accepts independent noisy maps, without fitting."""
import argparse
import json
import time
import zipfile
from pathlib import Path
import numpy as np
import jax
from frozen_detector import FrozenR3Detector
from common import sha, write_json


def main(a):
    out = Path(a.out)
    if out.exists(): raise FileExistsError(out)
    model = FrozenR3Detector(a.bundle)
    data = np.load(a.data, mmap_mode='r', allow_pickle=False)
    if not ((data.ndim == 2 and data.shape[1:] == (int(model.mask.sum()),)) or
            (data.ndim == 3 and data.shape[1:] == model.mask.shape)):
        raise ValueError('Input must be a batch of valid-pixel vectors or survey maps')
    result = np.empty(len(data), np.float64); start = time.monotonic()
    for lo in range(0, len(data), 16): result[lo:lo+16] = model.predict(data[lo:lo+16], a.variant)
    if not np.isfinite(result).all(): raise ValueError('Invalid scores')
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, 'x', zipfile.ZIP_DEFLATED) as z:
        z.writestr('result.json', json.dumps(dict(ood_scores=result.tolist()), allow_nan=False))
    write_json(str(out)+'.receipt.json', dict(variant=a.variant, count=len(result),
        archive_sha256=sha(out), input_sha256=sha(a.data),
        manifest_sha256=sha(Path(a.bundle)/'manifest.json'),
        backend=jax.default_backend(), jax_version=jax.__version__, seconds=time.monotonic()-start,
        model_or_test_fitting=False, submitted=False))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bundle','data','out'): p.add_argument('--'+name,required=True)
    p.add_argument('--variant',choices=['06','07'],required=True)
    main(p.parse_args())
