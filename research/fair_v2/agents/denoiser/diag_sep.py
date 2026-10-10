"""Diagnostic: per-feature separation (shift mean - audit mean)/audit sd for the den_r block. Synthetic sets only."""
import sys
from pathlib import Path

import numpy as np


def names_r():
    n = [f'rps{i}' for i in range(40)]
    for s in (0, 1, 2, 4):
        n += [f'rlogvar_s{s}', f'rskew_s{s}', f'rkurt_s{s}']
    n += [f'xcorr{i}' for i in range(20)]
    for s in (0, 1, 2, 4):
        n += [f'corr_ry_s{s}', f'corr_r2y_s{s}']
    n += ['sure_r2', 'div'] + [f'divbin{i}' for i in range(4)]
    for t in (8, 16):
        n += [f't{t}_logmu', f't{t}_cv', f't{t}_q05', f't{t}_q50', f't{t}_q95', f't{t}_corr_ym', f't{t}_corr_ys',
              f't{t}_lowk0', f't{t}_lowk1', f't{t}_lowk2']
    n += ['pos2.5', 'pos3', 'pos4', 'neg2.5', 'neg3', 'neg4']
    return n


def main():
    root = Path(sys.argv[1])
    a = np.load(root / 'audit.npz')['feats'][:, 0]
    names = names_r()
    nr = len(names)
    mu, sd = a.mean(0), a.std(0) + 1e-12
    for p in sorted(root.glob('shift_*.npz')):
        f = np.load(p)['feats'][:, 0]
        d = (f.mean(0) - mu) / sd
        dr = d[:nr]
        top = np.argsort(-np.abs(dr))[:8]
        sure = (f[:, 80] - 1 + 2 * f[:, 81]), (a[:, 80] - 1 + 2 * a[:, 81])
        print(f'{p.stem:22s} max|d| r-block={np.abs(dr).max():.2f} s-block={np.abs(d[nr:nr+466]).max():.2f} '
              f'b-block={np.abs(d[nr+466:]).max():.2f}  SURE d={(sure[0].mean()-sure[1].mean())/sure[1].std():.2f}')
        print('    ' + ' '.join(f'{names[i]}={dr[i]:+.2f}' for i in top))


if __name__ == '__main__':
    main()
