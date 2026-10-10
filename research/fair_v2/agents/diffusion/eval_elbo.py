"""Raw (unconditional) denoising-ELBO score: standardized on calib3, evaluated on audit3 + shifts."""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import phase2_score  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', type=Path, default=Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3/dife'))
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args()
    ld = lambda n: np.load(args.root / f'{n}.npz')['feats'][..., 0].reshape(-1)
    c = ld('calib')
    mu, sd = float(c.mean()), float(c.std())
    z = lambda n: (ld(n) - mu) / sd
    za = z('audit')
    shifts = sorted(p.stem for p in args.root.glob('shift_*.npz'))
    rep = {}
    for mode, f in (('upper', lambda v: v), ('two_sided', np.abs)):
        r = dict(normal_quantiles=np.quantile(f(za), [.5, .9, .95, .99, .995, .999]).round(3).tolist(),
                 shifts={s.removeprefix('shift_'): round(phase2_score(f(za), f(z(s))), 4) for s in shifts})
        r['shift_mean'] = round(float(np.mean(list(r['shifts'].values()))), 4)
        r['mean_shift_z'] = {s.removeprefix('shift_'): round(float(z(s).mean() - za.mean()), 3) for s in shifts}
        rep[mode] = r
        print(mode, json.dumps(r), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'elbo_raw.json').write_text(json.dumps(dict(calib_mean=mu, calib_std=sd, **rep), indent=1))
    if args.test:
        np.save(args.out / 'test_elbo_raw_z.npy', z('test'))


if __name__ == '__main__':
    main()
