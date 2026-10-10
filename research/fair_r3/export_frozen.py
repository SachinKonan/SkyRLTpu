"""Infer frozen candidates using verified cached field inference; validate raw API."""
import argparse
import json
import time
import zipfile
from pathlib import Path
import numpy as np
from frozen_detector import FrozenR3Detector
from common import sha, write_json


def main(a):
    root, out = Path(a.bundle), Path(a.out); out.mkdir(parents=True, exist_ok=False)
    detector = FrozenR3Detector(root); m = detector.manifest
    if sha(a.test) != m['public_test_sha256']: raise ValueError('Wrong public test version')
    if sha(a.field_cache) != m['frozen_field_cache_sha256']: raise ValueError('Wrong cached field inference')
    with np.load(a.field_cache, allow_pickle=False) as f: field = f['field_conditioned_score']
    data = np.load(a.test, mmap_mode='r', allow_pickle=False)
    assert data.shape == (10000, int(detector.mask.sum())) and field.shape == (10000,)
    assert np.isfinite(field).all()
    start = time.monotonic(); outputs = {k: [] for k in ('06','07')}
    for lo in range(0, len(data), 16):
        full = detector._full(data[lo:lo+16]); c = detector.summary_spatial(full)
        values = detector.combine(c['summary'], field[lo:lo+len(full)], c['regional'])
        for k, v in values.items(): outputs[k].append(v)
        if lo % 2048 == 0: print(json.dumps(dict(inferred=lo+len(full),seconds=time.monotonic()-start)),flush=True)
    outputs = {k: np.concatenate(v) for k, v in outputs.items()}
    # The full API independently recomputes field scores from individual raw maps.
    ids = np.asarray([0,1,17,113,509,1021,4321,9999]); x = np.asarray(data[ids])
    direct = detector.predict_both(x)
    alone = detector.predict_both(x[:1]); reverse = detector.predict_both(x[::-1])
    left = detector.predict_both(x[:3]); right = detector.predict_both(x[3:])
    full = np.full((len(x), *detector.mask.shape), np.nan, np.float32); full[:,detector.mask] = x
    masked = detector.predict_both(full)
    empty = detector.predict_both(np.empty((0,x.shape[1]),np.float32))
    failures = False
    try:
        bad = x[:1].copy(); bad[0,0] = np.nan; detector.predict_both(bad)
    except ValueError: failures = True
    assert failures and all(len(v)==0 for v in empty.values())
    checks = {}
    for k in outputs:
        comparisons = {'cached_field_vs_full_api': (outputs[k][ids],direct[k]),
                       'singleton': (direct[k][:1],alone[k]), 'reverse': (direct[k],reverse[k][::-1]),
                       'chunks': (direct[k],np.r_[left[k],right[k]]), 'unobserved_nan': (direct[k],masked[k])}
        checks[k] = {name: float(np.max(abs(v-w))) for name,(v,w) in comparisons.items()}
        for name, (v,w) in comparisons.items():
            if not np.allclose(v,w,rtol=2e-5,atol=1e-4): raise AssertionError((k,name,checks[k][name]))
    names = {'06':'06-noise-trained-hybrid.zip','07':'07-noise-hybrid-spatial.zip'}
    receipts = {}
    for k, values in outputs.items():
        assert values.shape == (10000,) and np.isfinite(values).all()
        path = out / names[k]
        with zipfile.ZipFile(path,'x',zipfile.ZIP_DEFLATED) as z:
            z.writestr('result.json',json.dumps(dict(ood_scores=values.tolist()),allow_nan=False))
        receipts[k] = dict(archive=path.name,sha256=sha(path),count=len(values),
            manifest_sha256=sha(root/'manifest.json'),public_test_sha256=m['public_test_sha256'],
            cached_field_sha256=sha(a.field_cache),full_api_check_indices=ids.tolist(),
            independence_max_abs_errors=checks[k],empty_and_nonfinite_guards=True,
            model_or_test_fitting=False,submitted=False,backend='TPU',seconds=time.monotonic()-start)
        write_json(out/(path.stem+'.receipt.json'),receipts[k])
    write_json(out/'complete.json',dict(receipts=receipts,seconds=time.monotonic()-start))
    print(json.dumps(receipts),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('bundle','test','field-cache','out'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
