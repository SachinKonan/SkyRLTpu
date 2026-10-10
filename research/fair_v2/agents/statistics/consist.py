"""Quick check of the stx families as independent theta-views (cross-family parameter consistency).

For each frozen r2 model (fit on fit3), theta_hat = profile argmin per map. For a set of views, the
disagreement vector D = [theta_v - theta_ref]_v (normalized by training theta std) is scored by its
Mahalanobis distance under calib3 normal maps (mean/cov of D on calib3), mapped chi2 -> normal score and
standardized on calib3. Per map only; calib3 for standardization; audit3 + synthetic shifts for evaluation.
usage: consist.py <ref> <view1,view2,...> [--test]
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats as sst

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from model import phase2_score  # noqa: E402
from gdirect import load  # noqa: E402

RUN = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/runs/r2')
V3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
DS = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/dshift')
ref, views = sys.argv[1], sys.argv[2].split(',')
TEST = '--test' in sys.argv
sets = ['calib', 'audit'] + sorted({p.stem for p in (V3 / 'stats').glob('shift_*.npz')} |
                                   {p.stem for p in (DS / 'stats').glob('shift_*.npz')}) + (['test'] if TEST else [])


def thetas(tag):
    p = RUN / f'theta_{tag}.npz'
    have = dict(np.load(p)) if p.exists() else {}
    if all(s in have for s in sets):
        return have
    d = pickle.load(open(RUN / f'model_{tag}.pkl', 'rb'))
    lk, spec = d['lk'], d['spec']
    for s in sets:
        if s in have:
            continue
        f = load(spec, s)
        f = f.reshape(-1, f.shape[-1])
        th = np.concatenate([lk.profile(f[lo:lo + 4096])[1] for lo in range(0, len(f), 4096)])
        have[s] = ((th - lk.t_mean) / lk.t_std).astype(np.float32)
    np.savez(p, **have)
    return have


T = {t: thetas(t) for t in [ref] + views}


def disagreement(s, vs):
    return np.concatenate([T[v][s] - T[ref][s] for v in vs], 1)


def score_fn(vs):
    Dc = disagreement('calib', vs)
    mu = Dc.mean(0); C = np.cov(Dc.T) + 1e-6 * np.eye(Dc.shape[1]); Ci = np.linalg.inv(C)
    k = Dc.shape[1]
    def raw(D):
        q = np.einsum('ni,ij,nj->n', D - mu, Ci, D - mu)
        return np.clip(sst.norm.ppf(np.clip(sst.chi2.cdf(q, k), 1e-300, 1 - 1e-16)), -8.3, 8.3)
    rc = raw(Dc); m, sd = rc.mean(), rc.std()
    return lambda s: (raw(disagreement(s, vs)) - m) / sd


base = {t: dict(np.load(RUN / f'z_{t}.npz')) for t in ('Sd', 'C')}
KEY = ['bary9.0', 'bary6.5', 'baryDMO', 'peakcut0.3', 'ia0.2', 'poisson120', 'psfpat', 'srcclust0.15', 'spread0.6_t2', 'concen0.6_t3']
std_shifts = [s for s in sets if s.startswith('shift_') and (V3 / 'stats' / f'{s}.npz').exists()]
for label, vs in [('all', views)] + [(v, [v]) for v in views]:
    fn = score_fn(vs)
    za = fn('audit')
    r = {s.removeprefix('shift_'): phase2_score(za, fn(s)) for s in sets if s.startswith('shift_')}
    ba = (base['Sd']['audit'] + base['C']['audit']) / 2
    rc = {}
    for s in sets:
        if s.startswith('shift_') and s in base['Sd']:
            rc[s.removeprefix('shift_')] = phase2_score((za + 2 * ba) / 3, (fn(s) + base['Sd'][s] + base['C'][s]) / 3)
    m = np.mean([r[s.removeprefix('shift_')] for s in std_shifts]); mc = np.mean([rc[s.removeprefix('shift_')] for s in std_shifts])
    print(f'[{ref} vs {label}] alone mean19={m:.4f} q99.9={np.quantile(za, .999):.2f} | +Sd+C mean19={mc:.4f} | ' +
          ' '.join(f'{k}={r.get(k, np.nan):.3f}/{rc.get(k, np.nan):.3f}' for k in KEY), flush=True)
    if TEST and label == 'all':
        np.save(RUN / f'test_consist_{ref}_vs_{"+".join(views)}.npy', fn('test'))
        np.save(RUN / f'test_consist_{ref}_vs_{"+".join(views)}_plus_Sd_C.npy',
                (fn('test') + base['Sd']['test'] + base['C']['test']) / 3)
