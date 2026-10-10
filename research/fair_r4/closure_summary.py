"""Frozen summary inference on the closure feature cache; no fitting."""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from detector import FrozenR4Detector
from common import write_json, sha


def main(a):
    out = Path(a.out); out.mkdir(parents=True, exist_ok=False)
    cache = Path(a.cache); f = np.load(cache/'features165.npy', allow_pickle=False)
    model = FrozenR4Detector(a.bundle); start = time.monotonic()
    result = model.summary_components(f)
    for key, value in result.items(): assert np.isfinite(value).all(), key
    check = model.summary_components(f[:31][::-1])
    error = {k: float(np.max(abs(v[:31]-check[k][::-1]))) for k, v in result.items()}
    assert max(error.values())<1e-4
    original = np.mean([fn(f[:31, :72].astype(np.float64))['minimum_base_energy']
                        for fn in model.base.summaries], axis=0)
    assert np.allclose(result['grid'][:31], original, rtol=1e-6, atol=1e-4)
    np.savez(out/'scores.npz', **result)
    report = dict(kind='FROZEN_CLOSURE_SUMMARY_INFERENCE', public_test_read=False, count=len(f),
                  seconds=time.monotonic()-start, pointwise_errors=error,
                  source_features_sha256=sha(cache/'features165.npy'),
                  bundle_manifest_sha256=sha(Path(a.bundle)/'r4-manifest.json'))
    write_json(out/'report.json', report); print(json.dumps(report), flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('cache','bundle','out'):p.add_argument('--'+name,required=True)
    main(p.parse_args())
