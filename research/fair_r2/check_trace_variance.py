"""Measure Hutchinson trace uncertainty using fixed NORMAL calibration maps.

No public data or proxy positives are accepted; this diagnoses the numerical
precision of the field score independently of model ranking.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
import jax
from field_evaluate import predictor
from common import sha, write_json


def main(a):
    cache = Path(a.cache)
    manifest = json.loads((cache/'manifest.json').read_text())
    if manifest['kind'] != 'NORMAL_ONLY_TRACE_AUDIT':
        raise ValueError('Expected normal calibration audit data')
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    maps = np.load(cache/'observed.npy', allow_pickle=False)
    start = time.monotonic()
    values = []
    for seed in (451, 452, 453, 454):
        fn = predictor(a.run, steps=32, probe_seed=seed, batch=32)
        result = fn(maps)
        assert np.isfinite(result['nll']).all()
        values.append(result['nll'])
        np.savez(out/f'probe-{seed}.npz', **result)
        print(json.dumps(dict(trace_seed=seed, count=len(maps),
                              seconds=time.monotonic()-start)), flush=True)
        del fn
        jax.clear_caches()
    values = np.asarray(values)
    spread = values.std(axis=0, ddof=1)
    report = dict(kind='NORMAL_ONLY_FIELD_TRACE_VARIANCE', count=len(maps),
                  cache_manifest_sha256=sha(cache/'manifest.json'),
                  model_sha256=sha(Path(a.run)/'best.msgpack'),
                  trace_seeds=[451,452,453,454], ode_steps=32,
                  median_trace_sd=float(np.median(spread)),
                  q90_trace_sd=float(np.quantile(spread,.9)),
                  max_trace_sd=float(np.max(spread)),
                  across_normal_maps_nll_sd=float(values.mean(0).std()),
                  rms_difference_from_seed451=float(np.sqrt(np.mean((values[1:]-values[:1])**2))),
                  seconds=time.monotonic()-start)
    write_json(out/'complete.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('run','cache','out'):
        p.add_argument('--'+name, required=True)
    main(p.parse_args())
