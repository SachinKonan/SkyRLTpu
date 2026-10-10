"""Combine standardized per-map z-scores from several fit_v3 runs (mean or max), report, and save test scores.

--z "D=/path/run1/z_D.npz,C=/path/run0/z_C.npz" --combos "mean:D+C,max:D+C" --out DIR [--test]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import phase2_score  # noqa: E402

# 'signal' shifts change the lensing field itself (physics-like); the rest change noise/systematics.
SIGNAL = ('mult', 'smooth', 'aniso', 'supp', 'boost', 'peakcut', 'bary', 'ia')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--z', required=True)
    ap.add_argument('--combos', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--test', action='store_true')
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    zs = {}
    for item in args.z.split(','):
        tag, path = item.split('=')
        with np.load(path) as f:
            zs[tag] = {k: f[k] for k in f.files}
    shifts = sorted(k for k in next(iter(zs.values())) if k.startswith('shift_'))
    report = {}
    for combo in args.combos.split(','):
        mode, _, body = combo.rpartition(':')
        parts = body.split('+')

        def comb(name):
            v = np.stack([zs[p][name] for p in parts])
            return v.max(0) if mode == 'max' else v.mean(0)
        za = comb('audit')
        res = dict(normal_quantiles=np.quantile(za, [.5, .9, .95, .99, .995, .999]).round(3).tolist(),
                   calib_q999=round(float(np.quantile(comb('calib'), .999)), 3),
                   shifts={s.removeprefix('shift_'): round(phase2_score(za, comb(s)), 4) for s in shifts})
        res['shift_mean'] = round(float(np.mean(list(res['shifts'].values()))), 4)
        sig = [k for k in res['shifts'] if k.split('0')[0].rstrip('+-.') in SIGNAL or k.startswith(SIGNAL)]
        oth = [k for k in res['shifts'] if k not in sig]
        res['signal_mean'] = round(float(np.mean([res['shifts'][k] for k in sig])), 4)
        res['noise_sys_mean'] = round(float(np.mean([res['shifts'][k] for k in oth])), 4)
        report[combo] = res
        print(f'[{combo}] nq={res["normal_quantiles"]} calib999={res["calib_q999"]} shift_mean={res["shift_mean"]} '
              f'signal={res["signal_mean"]} noise/sys={res["noise_sys_mean"]}')
        print('   ' + ' '.join(f'{k}={v:.3f}' for k, v in res['shifts'].items()), flush=True)
        if args.test:
            np.save(args.out / f'test_{combo.replace(":", "_")}.npy', comb('test'))
    (args.out / 'combos.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
