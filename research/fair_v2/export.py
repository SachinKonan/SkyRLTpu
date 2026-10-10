"""Write a Codabench Phase-2 result submission: zip containing result.json.

Scores are used exactly as computed per map (no rank transform, no test
normalization). Checks: exactly 10000 finite values.
"""
import argparse
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('scores', help='.npy with 10000 per-map OoD scores (higher = more OoD)')
    ap.add_argument('out', help='output .zip')
    args = ap.parse_args()
    s = np.asarray(np.load(args.scores), dtype=np.float64).ravel()
    if s.shape != (10000,) or not np.isfinite(s).all():
        raise SystemExit(f'bad scores: shape {s.shape}, finite {np.isfinite(s).all()}')
    out = Path(args.out)
    if out.exists():
        raise SystemExit(f'{out} exists')
    payload = json.dumps({'ood_scores': s.tolist()}, allow_nan=False, separators=(',', ':'))
    with zipfile.ZipFile(out, 'x', zipfile.ZIP_DEFLATED) as z:
        z.writestr('result.json', payload)
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    print(json.dumps(dict(out=str(out), count=len(s), sha256=digest,
                          quantiles=np.quantile(s, [0, .5, .9, .99, 1]).round(3).tolist())))


if __name__ == '__main__':
    main()
