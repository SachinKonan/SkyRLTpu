"""Normal-tail of the OE logit by parameter region (does it flag extreme-but-valid maps?).

L = logit standardized on calib3 (mean/std). For each parameter, maps are binned
into quintiles plus the extreme 5% at each end; per bin we report the fraction of
maps above the calib3 99% and 99.9% thresholds (expect 1% / 0.1%) and the bin max.
Uses calib3 (2 views) and audit3 (1 view) InD maps, and optionally fit_v3 z-scores
(--run, e.g. model O / combos) for the same maps.
"""
import argparse
import json
from pathlib import Path

import numpy as np

NAMES = ['Omega_m', 'S8', 'T_AGN', 'f0', 'Delta_z']


def table(z_cal, lab_cal, z_aud, lab_aud):
    t99, t999 = np.quantile(z_cal, [.99, .999])
    res = {}
    for j, nm in enumerate(NAMES):
        edges = np.quantile(np.r_[lab_cal[:, j], lab_aud[:, j]], [0, .05, .2, .4, .6, .8, .95, 1])
        rows = []
        for a, b in zip(edges[:-1], edges[1:]):
            sel_c = (lab_cal[:, j] >= a) & (lab_cal[:, j] <= b)
            sel_a = (lab_aud[:, j] >= a) & (lab_aud[:, j] <= b)
            zc, za = z_cal[sel_c], z_aud[sel_a]
            rows.append(dict(range=[round(float(a), 4), round(float(b), 4)], n_cal=int(sel_c.sum()), n_aud=int(sel_a.sum()),
                             cal_gt99=round(float((zc > t99).mean()), 4), aud_gt99=round(float((za > t99).mean()), 4),
                             aud_gt999=round(float((za > t999).mean()), 4), aud_max=round(float(za.max()), 2)))
        res[nm] = rows
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
    ap.add_argument('--src', default='oe')
    ap.add_argument('--run', default='')
    ap.add_argument('--tags', default='O')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    root = Path(args.root) / args.src
    c = np.load(root / 'calib.npz'); a = np.load(root / 'audit.npz')
    fc, fa = c['feats'], a['feats']
    lab_cal = np.repeat(c['labels'], fc.shape[1], axis=0)
    lab_aud = a['labels']
    lc, la = fc[..., 0].reshape(-1).astype(np.float64), fa[..., 0].reshape(-1).astype(np.float64)
    mu, sd = lc.mean(), lc.std()
    out = {'L': table((lc - mu) / sd, lab_cal, (la - mu) / sd, lab_aud)}
    if args.run:
        for tag in args.tags.split(','):
            z = np.load(Path(args.run) / f'z_{tag}.npz')
            out[tag] = table(z['calib'], lab_cal, z['audit'], lab_aud)
    Path(args.out).write_text(json.dumps(out, indent=1))
    for k, v in out.items():
        print(f'== {k}: fraction of audit3 maps above calib 99% threshold, per bin [5%,20%,..,95%,100%]')
        for nm, rows in v.items():
            print(f'  {nm:8s} aud>99: ' + ' '.join(f'{r["aud_gt99"]:.3f}' for r in rows)
                  + ' | cal>99: ' + ' '.join(f'{r["cal_gt99"]:.3f}' for r in rows)
                  + ' | aud max: ' + ' '.join(f'{r["aud_max"]:.1f}' for r in rows))


if __name__ == '__main__':
    main()
