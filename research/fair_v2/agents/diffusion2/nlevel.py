"""Large-vs-small-scale theta view from the diffusion features, split by NOISE LEVEL (resumable).

difa columns per level k (sigma_t = LEVELS[k]): [e-bands 8 if sigma_t>0] + r-bands 8 + kh-bands 8 + kh mom 2 + r mom 2.
Families (each fitted with fit_v3 settings on fit3, profiled on every set incl. test):
  nlo : all difa columns of levels 0-3 (sigma_t <= 0.04, denoiser keeps small scales)
  nhi : all difa columns of levels 4-7 (sigma_t >= 0.08, denoiser reconstructs only large scales)
  LS  : kh (posterior-mean map) band powers + kh moments at levels 5-7 (pure large-scale reconstruction)
  SS  : ELBO-error bands of levels 1-3 + residual (x - kh) bands/moments of levels 0-2 (small-scale structure)
Then: cross copula energies (A at B's theta) for the pairs (nlo,nhi) and (SS,LS) -> xNL, xSSLS (like xSL),
theta-disagreement Mahalanobis cons scores, standardized on calib3 like fit_v3. Saves z_<tag>.npz and fam/<tag>.npz.
"""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import Likelihood, Standardizer, phase2_score  # noqa: E402

ROOT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
OUT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion2/nlevel')
LEVELS = [0.0, 0.01, 0.02, 0.04, 0.08, 0.16, 0.32, 0.64]


def start(k):
    return 0 if k == 0 else 20 + (k - 1) * 28


def lvl(k):
    """dict of column blocks of difa for level k."""
    s = start(k)
    e = list(range(s, s + 8)) if k > 0 else []
    o = s + len(e)
    return dict(e=e, r=list(range(o, o + 8)), kh=list(range(o + 8, o + 16)), khm=[o + 16, o + 17],
                rm=[o + 18, o + 19])


def fam_cols():
    nlo = sum([sum(lvl(k).values(), []) for k in range(4)], [])
    nhi = sum([sum(lvl(k).values(), []) for k in range(4, 8)], [])
    LS = sum([lvl(k)['kh'] + lvl(k)['khm'] for k in (5, 6, 7)], [])
    SS = sum([lvl(k)['e'] for k in (1, 2, 3)], []) + sum([lvl(k)['r'] + lvl(k)['rm'] for k in (0, 1, 2)], [])
    assert nlo + nhi == list(range(216)), (len(nlo), len(nhi))
    return dict(nlo=np.array(nlo), nhi=np.array(nhi), LS=np.array(LS), SS=np.array(SS))


def main():
    t0 = time.time()
    dev = 'cuda'
    (OUT / 'fam').mkdir(parents=True, exist_ok=True)
    shifts = sorted(p.stem for p in (ROOT / 'stats').glob('shift_*.npz'))
    sets = ['calib', 'audit'] + shifts + ['test']
    F = {s: np.load(ROOT / 'difa' / f'{s}.npz')['feats'] for s in ['fit'] + sets}
    lab = np.load(ROOT / 'stats' / 'fit.npz')['labels']
    FC = fam_cols()
    LK = {}
    for tag, c in FC.items():
        fp = OUT / 'fam' / f'{tag}.npz'
        if fp.exists():
            continue
        lk = Likelihood(dev, 0.02).fit(F['fit'][..., c], lab, steps=3000, width=256, log=lambda m: None, jitter=0.1)
        res = {}
        for s in sets:
            f = F[s][..., c]
            n, v, d = f.shape
            _, th = lk.profile(f.reshape(n * v, d))
            res[f'{s}/th'] = th.astype(np.float32); res[f'{s}/cop'] = lk.last_copula
        LK[tag] = lk
        np.savez(fp, **res)
        torch.save(lk, OUT / 'fam' / f'model_{tag}.pt')
        print(f'fitted {tag} d={len(c)} {time.time()-t0:.0f}s', flush=True)
    # cross energies
    for a, b in (('nlo', 'nhi'), ('SS', 'LS')):
        fp = OUT / 'fam' / f'cross_{a}_{b}.npz'
        if fp.exists():
            continue
        lks = {t: LK.get(t) or torch.load(OUT / 'fam' / f'model_{t}.pt', weights_only=False) for t in (a, b)}
        th = {t: np.load(OUT / 'fam' / f'{t}.npz') for t in (a, b)}
        res = {}
        for s in sets:
            for f, g in ((a, b), (b, a)):
                x = F[s][..., FC[f]]
                n, v, d = x.shape
                xs = lks[f].standardize(x.reshape(n * v, d))
                t = torch.as_tensor((th[g][f'{s}/th'] - lks[f].t_mean) / lks[f].t_std, dtype=torch.float32,
                                    device=dev)
                with torch.no_grad():
                    res[f'{s}/cop_{f}@{g}'] = lks[f].copula_energy(xs, t).cpu().numpy().astype(np.float64)
        np.savez(fp, **res)
        print(f'cross {a} {b} {time.time()-t0:.0f}s', flush=True)
    print('ALL DONE', flush=True)


if __name__ == '__main__':
    main()
