"""Stand-alone OE logit score + combinations with fit_v3 detector z-scores.

L  : OE logit standardized on calib3 (location/scale constant).
Lc : OE logit standardized on calib3 as a quadratic function of the network's own
     parameter predictions (model.Standardizer), i.e. per-map, frozen.
Combos read fit_v3 outputs (z_<tag>.npz) from --run and combine standardized
z per map (mean or max), e.g. --combos "mean:Lc+Sd+C,max:Lc+C".
Writes report json + per-map test z arrays (only with --test).
"""
import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import numpy as np

import data as D
from model import Standardizer, phase2_score


def one(f, col):
    N, V, d = f.shape
    f = f.reshape(N * V, d)
    nfh = d - 1 - 5 - 256
    if col == -1:     # family-head "any OoD" score: logsumexp(OoD logits) - InD logit
        fl = f[:, 1:1 + nfh]
        mx = fl[:, 1:].max(1, keepdims=True)
        r = (mx[:, 0] + np.log(np.exp(fl[:, 1:] - mx).sum(1))) - fl[:, 0]
    else:
        r = f[:, col].astype(np.float64)
    return r, f[:, 1 + nfh:6 + nfh].astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
    ap.add_argument('--src', default='oe')
    ap.add_argument('--run', default='')
    ap.add_argument('--combos', default='')
    ap.add_argument('--col', type=int, default=0)
    ap.add_argument('--out', required=True)
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--extra', default='', help='TAG=path/z_X.npz,... extra per-map z sources (fit_v3 outputs)')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    srcs = args.src.split('+')        # several OE sources: average their raw scores (ensemble)
    names = ['calib', 'audit'] + [f'shift_{s}' for s in D.SHIFTS] + (['test'] if args.test else [])
    raw, th = {}, {}
    for n in names:
        rs = []
        for src in srcs:
            f = np.load(Path(args.root) / src / f'{n}.npz')['feats']
            rs.append(one(f, args.col))
        raw[n] = np.mean([r[0] for r in rs], 0)
        th[n] = rs[0][1]
    mu, sd = raw['calib'].mean(), raw['calib'].std()

    stdz = Standardizer().fit(raw['calib'], th['calib'], np.zeros(5), np.ones(5))
    zs = {'L': {n: (raw[n] - mu) / sd for n in names}, 'Lc': {n: stdz(raw[n], th[n]) for n in names}}
    if args.run:
        for p in Path(args.run).glob('z_*.npz'):
            with np.load(p) as f:
                zs[p.stem[2:]] = {k: f[k] for k in f.files}
    for kv in filter(None, args.extra.split(',')):
        tag, path = kv.split('=')
        with np.load(path) as f:
            zs[tag] = {k: f[k] for k in f.files}
    shifts = [n for n in names if n.startswith('shift_')]
    report = {}

    def evaluate(tag, get):
        za = get('audit')
        res = dict(normal_quantiles=np.quantile(za, [.5, .9, .95, .99, .995, .999]).round(3).tolist(),
                   normal_max=round(float(za.max()), 2),
                   shifts={s.removeprefix('shift_'): round(phase2_score(za, get(s)), 4) for s in shifts})
        res['shift_mean'] = round(float(np.mean(list(res['shifts'].values()))), 4)
        report[tag] = res
        print(f'[{tag}] q={res["normal_quantiles"]} max={res["normal_max"]} shift_mean={res["shift_mean"]}', flush=True)
        print('   ' + ' '.join(f'{k}={v:.3f}' for k, v in res['shifts'].items()), flush=True)
        if args.test:
            np.save(out / f'test_{tag.replace(":", "_")}.npy', get('test'))

    for tag in ('L', 'Lc'):
        evaluate(tag, lambda n, t=tag: zs[t][n])
    for combo in filter(None, args.combos.split(',')):
        mode, _, body = combo.rpartition(':')
        parts = body.split('+')

        def get(n, parts=parts, mode=mode):
            v = np.stack([zs[p][n] for p in parts])
            return v.max(0) if mode == 'max' else v.mean(0)
        evaluate('combo:' + combo, get)
    (out / 'score_report.json').write_text(json.dumps(report, indent=1))


if __name__ == '__main__':
    main()
