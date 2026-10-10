"""CPU analysis of the OE members (no fitting on test; test z written only as frozen per-map outputs).

Scores
  L_<src> : OE classifier logit (column 0) standardized on calib3 (constant mean/std).
  O, Oh   : fit_v3 likelihood z on oe:raw / oe_head:raw (runs/r1).
  v3-04 components (Sd, C, Dr, xSL, consP8) from agents/consistency/export3/components_v3-04_repro.npz.
Reports audit3 normal tail, per-shift phase2 score, tail by parameter region, and combos with v3-04.
Writes z_<tag>.npz (calib/audit/shift_*/test) for the L scores and chosen combos into --out.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))

import data as D  # noqa: E402
from model import phase2_score  # noqa: E402

ROOT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
COMP = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency/export3/components_v3-04_repro.npz')
SHIFTS = list(D.SHIFTS)
SETS = ['calib', 'audit'] + ['shift_' + s for s in SHIFTS] + ['test']
NAMES = ['Omega_m', 'S8', 'T_AGN', 'f0', 'Delta_z']
PHYS = ['bary6.5', 'bary9.0', 'baryDMO', 'peakcut0.3', 'ia0.2', 'supp0.15', 'boost0.15']


def L_of(src):
    raw = {}
    for n in SETS:
        p = ROOT / src / f'{n}.npz'
        if not p.exists():
            return None
        f = np.load(p)['feats']
        raw[n] = f[..., 0].reshape(-1).astype(np.float64)
    mu, sd = raw['calib'].mean(), raw['calib'].std()
    return {n: (v - mu) / sd for n, v in raw.items()}


def evaluate(z):
    za = z['audit']
    sh = {s: phase2_score(za, z['shift_' + s]) for s in SHIFTS}
    return dict(q=np.quantile(za, [.5, .9, .99, .995, .999]).round(2).tolist(), mx=round(float(za.max()), 1),
                cq=np.quantile(z['calib'], [.99, .999]).round(2).tolist(),
                m19=float(np.mean(list(sh.values()))), phys=float(np.mean([sh[s] for s in PHYS])), sh=sh)


def region(z, lab_cal, lab_aud):
    t99, t999 = np.quantile(z['calib'], [.99, .999])
    out = {}
    for j, nm in enumerate(NAMES):
        edges = np.quantile(np.r_[lab_cal[:, j], lab_aud[:, j]], [0, .05, .2, .4, .6, .8, .95, 1])
        rows = []
        for a, b in zip(edges[:-1], edges[1:]):
            sa = (lab_aud[:, j] >= a) & (lab_aud[:, j] <= b)
            sc = (lab_cal[:, j] >= a) & (lab_cal[:, j] <= b)
            both = np.r_[z['audit'][sa], z['calib'][sc]]
            rows.append(dict(rng=[float(a), float(b)], n=int(len(both)), gt99=float((both > t99).mean()),
                             gt999=float((both > t999).mean()), aud_gt99=float((z['audit'][sa] > t99).mean()),
                             aud_max=float(z['audit'][sa].max())))
        out[nm] = rows
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier2/runs/r1')
    ap.add_argument('--out', default='/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier2/runs/L')
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    zs = {}
    with np.load(COMP) as f:
        for k in f.files:
            m, s = k.split('/')
            zs.setdefault('v4' + m, {})[s] = f[k]
    for t in ('O', 'Oh', 'Dr', 'C', 'Sd'):
        p = Path(args.run) / f'z_{t}.npz'
        if p.exists():
            with np.load(p) as f:
                zs[t] = {k: f[k] for k in f.files}
    for src in ('oe', 'oe_ho', 'oe_phys', 'oe_s1'):
        L = L_of(src)
        if L is not None:
            zs['L_' + src] = L
    Ls = [k for k in zs if k.startswith('L_') and k != 'L_oe_ho']
    if len(Ls) > 1:
        zs['L_ens'] = {n: np.mean([zs[k][n] for k in Ls], 0) for n in SETS}
    v4 = ['v4Sd', 'v4C', 'v4Dr', 'v4xSL', 'v4consP8']

    def combo(w):
        tot = sum(w.values())
        return {n: sum(wi * zs[k][n] for k, wi in w.items()) / tot for n in SETS}
    base = {k: 1.0 for k in v4}
    combos = {'v3-04': base, 'v3-04+O': {**base, 'O': 1.0}, 'v3-04+0.5O': {**base, 'O': 0.5},
              'v3-04+Oh': {**base, 'Oh': 1.0}, 'v3-04+L_oe': {**base, 'L_oe': 1.0},
              'v3-04+0.5L_oe': {**base, 'L_oe': 0.5}, 'v3-04+L_oe_ho': {**base, 'L_oe_ho': 1.0},
              'fit:mean:O+Dr+Sd+C': {'O': 1, 'Dr': 1, 'Sd': 1, 'C': 1}, 'fit:mean:Dr+Sd+C': {'Dr': 1, 'Sd': 1, 'C': 1},
              'fit:mean:Oh+Dr+Sd+C': {'Oh': 1, 'Dr': 1, 'Sd': 1, 'C': 1}}
    if 'L_ens' in zs:
        combos['v3-04+L_ens'] = {**base, 'L_ens': 1.0}
        combos['v3-04+0.5L_ens'] = {**base, 'L_ens': 0.5}
    for name, w in combos.items():
        if all(k in zs for k in w):
            zs[name] = combo(w)
    lab_cal = np.repeat(np.load(ROOT / 'stats' / 'calib.npz')['labels'], 2, axis=0)
    lab_aud = np.load(ROOT / 'stats' / 'audit.npz')['labels']
    report = {}
    print(f'{"score":22s} {"q50/90/99/99.5/99.9":28s} {"max":>5s} {"m19":>6s} {"phys7":>6s} | per shift')
    for tag, z in zs.items():
        r = evaluate(z)
        r['region'] = region(z, lab_cal, lab_aud)
        report[tag] = r
        print(f'{tag:22s} {"/".join(f"{v:.2f}" for v in r["q"]):28s} {r["mx"]:5.1f} {r["m19"]:.4f} {r["phys"]:.4f} | '
              + ' '.join(f'{s}={v:.3f}' for s, v in r['sh'].items()), flush=True)
    print('\n== tail by region: fraction of calib3+audit3 maps above calib 99% (expect 0.010) for bins '
          '[0-5,5-20,20-40,40-60,60-80,80-95,95-100%] of each parameter; then above 99.9%')
    for tag in [t for t in report if not t.startswith('v4') or t == 'v3-04']:
        r = report[tag]['region']
        print(f'-- {tag}')
        for nm in ('T_AGN', 'f0', 'Delta_z', 'Omega_m', 'S8'):
            rows = r[nm]
            print(f'   {nm:8s} >99: ' + ' '.join(f'{x["gt99"]:.3f}' for x in rows)
                  + '  >99.9: ' + ' '.join(f'{x["gt999"]:.4f}' for x in rows)
                  + '  aud max: ' + ' '.join(f'{x["aud_max"]:.1f}' for x in rows))
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    for tag in [t for t in zs if t.startswith('L_') or t.startswith('v3-04')]:
        np.savez(out / f'z_{tag}.npz', **zs[tag])
    print('wrote', out)


if __name__ == '__main__':
    main()
