"""Leaderboard files and ready-to-submit packages for config-only variants of a built package.

  python make_pkg_variants.py [variant ...]      (default: all; existing records are kept for the rest)

The member outputs come from ONE run of the package's own Model().predict on the public test file (verify_full.py
--mode full, stored per component in _verify/<base>/full_a100.npz); each variant combines them exactly as
model.py does: components with weight 0 are skipped, score = sum(w * z) / sum(w), nan/inf mapped to +-1e6.
Every variant package is a copy of the base package that differs only in config.json.
Test maps are only scored with frozen models; nothing is fit, calibrated or selected on them.
"""
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

DATA = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data')
PKG = DATA / 'packages'
BASE = 'v3-13a'
RUN = PKG / '_verify' / BASE / 'full_a100.npz'
VARIANTS = {
    'v3-15a': ('pkg-v313a', {}),                          # as packaged, including the package's own SS
    'v3-15b': ('pkg-v313a-noSS', {'SS': 0.0}),
    'v3-15c': ('pkg-v313b-noSS', {'SS': 0.0, 'Oh': 0.0}),
    # after v3-15b scored 0.6667: lower base-likelihood share / heavier consistency (earlier LB pattern)
    'v3-15d': ('pkg-v315b-noC', {'SS': 0.0, 'C': 0.0}),
    'v3-15e': ('pkg-v315b-cons4.5', {'SS': 0.0, 'consRplus': 4.5}),
    'v3-15f': ('pkg-v315b-noC-cons4.5', {'SS': 0.0, 'C': 0.0, 'consRplus': 4.5}),
    # after v3-15d (C 0) scored 0.6676: does Oh help on the real test? (its LCO gain is generator-overlap inflated)
    'v3-15g': ('pkg-v315d-noOh', {'SS': 0.0, 'C': 0.0, 'Oh': 0.0}),
}


def combine(z, weights):
    """model.Model.predict's combination step."""
    w = {k: float(v) for k, v in weights.items() if float(v) != 0.0}
    s = np.sum([v * np.asarray(z[k], dtype=np.float64) for k, v in w.items()], 0) / np.sum(list(w.values()))
    return np.nan_to_num(s, nan=1e6, posinf=1e6, neginf=-1e6), w


def write_result_zip(s, out):
    """Same writer as export.py: one result.json with 10000 finite scores, deflated."""
    s = np.asarray(s, dtype=np.float64).ravel()
    assert s.shape == (10000,) and np.isfinite(s).all()
    payload = json.dumps({'ood_scores': s.tolist()}, allow_nan=False, separators=(',', ':'))
    if out.exists():
        out.unlink()
    with zipfile.ZipFile(out, 'x', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('result.json', payload)
    with zipfile.ZipFile(out) as zf:                     # format check on the written file
        assert zf.namelist() == ['result.json']
        back = np.asarray(json.loads(zf.read('result.json'))['ood_scores'], dtype=np.float64)
    assert back.shape == (10000,) and np.isfinite(back).all() and np.array_equal(back, s)
    return hashlib.sha256(out.read_bytes()).hexdigest()


def top_overlap(a, b, k=100):
    return len(set(np.argsort(-a)[:k]) & set(np.argsort(-b)[:k]))


def main(only=None):
    run = np.load(RUN)
    z = {k[2:]: run[k] for k in run.files if k.startswith('z/')}
    base_cfg = json.loads((PKG / BASE / 'config.json').read_text())
    s_base, _ = combine(z, base_cfg['weights'])
    assert np.array_equal(s_base, run['scores']), 'combination does not reproduce the package run'
    refs = {n: np.load(DATA / 'merged' / f'{n}.npy') for n in ('v3-13a', 'v3-13b')}
    if (DATA / 'merged' / 'v3-15b.npy').exists():
        refs['v3-15b'] = np.load(DATA / 'merged' / 'v3-15b.npy')                 # LB 0.6667
    report = dict(source_run=str(RUN), run_seconds=float(run['seconds']), variants={})
    out_json = PKG / '_verify' / 'v3-15_variants.json'
    if only and out_json.exists():
        report['variants'] = json.loads(out_json.read_text())['variants']      # keep the earlier variants' records
    for name, (label, override) in VARIANTS.items():
        if only and name not in only:
            continue
        weights = dict(base_cfg['weights'], **override)
        s, used = combine(z, weights)
        np.save(DATA / 'merged' / f'{name}.npy', s)
        sub = DATA / 'submissions' / f'{name}-{label}.zip'
        sub_sha = write_result_zip(s, sub)
        # package: a copy of the base package with only config.json changed
        d = PKG / name
        if d.exists():
            shutil.rmtree(d)
        shutil.copytree(PKG / BASE, d)
        cfg = dict(base_cfg, weights=weights)
        (d / 'config.json').write_text(json.dumps(cfg, indent=1))
        zp = PKG / f'{name}.zip'
        if zp.exists():
            zp.unlink()
        shutil.make_archive(str(d), 'zip', root_dir=d)
        with zipfile.ZipFile(zp) as zf:
            names = set(zf.namelist())
        assert 'model.py' in names and names == {p.name for p in d.iterdir()}
        pkg_sha = hashlib.sha256(zp.read_bytes()).hexdigest()
        r = dict(label=label, weights_used=used, submission=str(sub), submission_sha256=sub_sha,
                 package=str(zp), package_sha256=pkg_sha, package_mb=round(zp.stat().st_size / 1e6, 1))
        for rn, rv in refs.items():
            r[f'spearman_vs_{rn}'] = round(float(spearmanr(s, rv).correlation), 5)
            r[f'top100_vs_{rn}'] = top_overlap(s, rv)
        r['quantiles'] = np.quantile(s, [0, .5, .9, .99, 1]).round(3).tolist()
        report['variants'][name] = r
        print(json.dumps({name: r}), flush=True)
    out_json.write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    import sys
    main(set(sys.argv[1:]) or None)       # e.g. `python make_pkg_variants.py v3-15d v3-15e v3-15f`
