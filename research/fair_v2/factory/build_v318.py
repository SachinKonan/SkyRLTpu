"""v3-18 = v3-16b with learned combiner weights (lever 1), config-only.

  python factory/build_v318.py '<json weights over Sd,Dr,xSL,consP8,consRplus,N2,PsBh,Oh,X,Ol>' <label>

Same procedure as factory/build_v316c.py: the leaderboard file combines the per-component test outputs of v3-16b's
own full package run (_verify/v3-16b/full_gh.npz) with model.py's combine (a check first reproduces v3-16b's
stored scores); the package is a copy of packages/v3-16b with only config.json's weights changed (no __pycache__).
"""
import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from make_pkg_variants import combine, write_result_zip, top_overlap  # noqa: E402

DATA = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data')
PKG = DATA / 'packages'
NAME = 'v3-18'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    new = {('X' if k == 'Xdp' else k): float(v) for k, v in json.loads(sys.argv[1]).items()}
    label = sys.argv[2]
    run = np.load(PKG / '_verify' / 'v3-16b' / 'full_gh.npz')
    z = {k[2:]: run[k] for k in run.files if k.startswith('z/')}
    base = json.loads((PKG / 'v3-16b' / 'config.json').read_text())
    s_b, _ = combine(z, base['weights'])
    assert np.allclose(s_b, run['scores'], atol=1e-4), np.abs(s_b - run['scores']).max()
    assert set(new) <= set(z), set(new) - set(z)
    w = dict(base['weights'])
    for k in w:
        if k in z:
            w[k] = new.get(k, 0.0)
    s, used = combine(z, w)
    sub = DATA / 'submissions' / f'{NAME}-pkg-{label}.zip'
    write_result_zip(s, sub)
    dst = PKG / NAME
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(PKG / 'v3-16b', dst, ignore=shutil.ignore_patterns('__pycache__'))
    (dst / 'config.json').write_text(json.dumps(dict(base, weights=w), indent=1))
    zp = PKG / f'{NAME}.zip'
    with zipfile.ZipFile(zp, 'w', zipfile.ZIP_DEFLATED) as zf:
        for f in sorted(dst.rglob('*')):
            if f.is_file() and '__pycache__' not in f.parts:
                zf.write(f, f.relative_to(dst))
    with zipfile.ZipFile(sub) as zf:
        back = np.asarray(json.loads(zf.read('result.json'))['ood_scores'], float)
    assert len(back) == 10000 and np.isfinite(back).all() and np.allclose(back, s)
    names_b = set(zipfile.ZipFile(PKG / 'v3-16b.zip').namelist())
    rep = dict(weights_used=used, submission=str(sub), submission_sha256=sha(sub), package=str(zp),
               package_sha256=sha(zp), same_files_as_v3_16b=set(zipfile.ZipFile(zp).namelist()) == names_b,
               spearman_vs_v3_16b=float(spearmanr(s, s_b)[0]), top100_vs_v3_16b=int(top_overlap(s, s_b)))
    (PKG / '_verify' / f'{NAME}_build.json').write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == '__main__':
    main()
