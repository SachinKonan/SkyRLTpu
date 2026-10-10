"""Scale-consistency scores (Dai & Seljak style dlogp) built from scalecons.py outputs (CPU)."""
import argparse

import numpy as np

import lib
from analyze import ALL, standardize, ev, mean_of, get, fam, boot_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--boot', default='')
    ap.add_argument('--export', default='')
    args = ap.parse_args()
    with np.load(lib.OUT / 'scale' / 'scale.npz') as d:
        X = {k: d[k] for k in d.files}
    sets = [s for s in ALL if f'{s}/th_small' in X]
    REG = {}

    def report(name, z):
        REG[name] = z
        print(lib.fmt(name, ev(z)), flush=True)

    F = {f: fam(f) for f in ('dog', 'cnn')}
    zSd = standardize({s: F['dog'][f'{s}/cop'] for s in sets}, {s: F['dog'][f'{s}/th'] for s in sets})
    zC = standardize({s: F['cnn'][f'{s}/cop'] for s in sets}, {s: F['cnn'][f'{s}/th'] for s in sets})
    base = mean_of(zSd, zC)
    report('BASE mean:Sd+C', base)
    srcs = sorted({k.split('@')[1] for k in X if k.startswith('audit/E_small@')})
    print('sources:', srcs)
    cands = {}
    for kind in ('E', 'cop'):
        for f in ('small', 'large', 'mid', 'cnn'):
            if f'audit/{kind}_{f}@{f}' not in X:
                continue
            for src in srcs:
                if src == f or f'audit/{kind}_{f}@{src}' not in X:
                    continue
                raw = {s: X[f'{s}/{kind}_{f}@{src}'] - X[f'{s}/{kind}_{f}@{f}'] for s in sets}
                if np.std(raw['calib']) < 1e-6 or src in ('r1Sd', 'r1C'):
                    continue  # identical model (dog==r1 Sd, cnn==r1 C)
                th = {s: X[f'{s}/th_{src}'] for s in sets}
                z = standardize(raw, th)
                name = f'dlogp[{kind}] {f}|{src}'
                cands[name] = z
                report(name, z)
    # two-directional (small given large AND large given small) and joint tensions
    for kind in ('E', 'cop'):
        for src in ('jointS', 'jointSC', 'r1S', 'r1SC'):
            if f'audit/{kind}_small@{src}' not in X:
                continue
            raw = {s: sum(X[f'{s}/{kind}_{f}@{src}'] - X[f'{s}/{kind}_{f}@{f}'] for f in ('small', 'large'))
                   for s in sets}
            z = standardize(raw, {s: X[f'{s}/th_{src}'] for s in sets})
            cands[f'dlogp[{kind}] small+large|{src}'] = z
            report(f'dlogp[{kind}] small+large|{src}', z)
        raw = {s: (X[f'{s}/{kind}_small@large'] - X[f'{s}/{kind}_small@small']) +
               (X[f'{s}/{kind}_large@small'] - X[f'{s}/{kind}_large@large']) for s in sets}
        z = standardize(raw, {s: 0.5 * (X[f'{s}/th_small'] + X[f'{s}/th_large']) for s in sets})
        cands[f'dlogp[{kind}] cross small<->large'] = z
        report(f'dlogp[{kind}] cross small<->large', z)
    print('--- with BASE')
    rows = []
    for name, z in cands.items():
        for w in (0.5, 1.0):
            zz = mean_of(base, z, w=[2, w])  # = mean(Sd, C, w*new) up to scale
            r = ev(zz)
            REG[f'BASE+{w}*[{name}]'] = zz
            rows.append((r['shift_mean'], f'BASE+{w}*[{name}]', r))
    for m, name, r in sorted(rows, key=lambda x: -x[0])[:25]:
        print(lib.fmt(name, r))
    if args.boot:
        print('--- paired bootstrap vs BASE')
        for name in args.boot.split(';'):
            m, sd, pp = boot_diff(REG[name], None, base, None)
            print(f'  {name:50s} diff={m:+.4f} +- {sd:.4f}  P(>0)={pp:.2f}', flush=True)
    np.savez(lib.OUT / 'scale' / 'z_registry_keys.npz', keys=np.array(list(REG)))
    for name in filter(None, args.export.split(';')):
        z = REG[name]
        slug = ''.join(c if c.isalnum() or c in '+-.' else '_' for c in name)
        d = lib.OUT / 'export'; d.mkdir(exist_ok=True)
        assert z['test'].shape == (10000,) and np.isfinite(z['test']).all()
        np.save(d / f'test_z_{slug}.npy', z['test'].astype(np.float64))
        np.savez(d / f'z_{slug}.npz', **z)
        print('exported', name, '->', d / f'test_z_{slug}.npy')


if __name__ == '__main__':
    main()
