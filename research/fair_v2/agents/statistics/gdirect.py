"""Targeted (directional) tests in the residual space of the conditional likelihood.

The omnibus profile score g^T Rc^-1 g (g = copula normal scores of the residuals at the profiled theta)
pays a dimension penalty ~ sqrt(2p). Here we keep the same theta-conditioned residuals g but test only
K physically motivated directions: for synthetic shift k applied to fit3 maps (tshift/, training maps
only), delta_k = mean paired change of g; projection s_k = delta_k^T Rc^-1 g (the LR direction for a mean
shift delta_k). z_k standardized on calib3, T = z^T C^-1 z, chi2_K -> normal score -> theta-conditional
Standardizer on calib3. Leave-family-out evaluation on audit3 shifts. All per map.
"""
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from scipy import stats as sst

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from groups import groups  # noqa: E402
from model import Likelihood, Standardizer, phase2_score  # noqa: E402

V3 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
TS = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/tshift')
DS = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/dshift')
RUNS = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/runs')
FAMS = {'baryS': ['bary9.0', 'supp0.15'], 'baryW': ['baryDMO', 'bary6.5', 'boost0.15'],
        'spread': ['spread0.6_t2', 'spread0.3_t2'], 'concen': ['concen0.6_t3', 'concen0.3_t2'],
        'ia': ['ia0.2'], 'peakcut': ['peakcut0.3'], 'psf': ['psfpat'], 'srcclust': ['srcclust0.15']}
G, _ = groups()


def cols_of(subset, d):
    if subset == 'raw':
        return np.arange(d)
    if subset[0] in 'xk' and subset[1:2].isdigit():
        sel = []
        for tok in subset[1:].split('.'):
            a, _, b = tok.partition('-'); sel += list(range(int(a), int(b or a) + 1))
        return np.setdiff1d(np.arange(d), sel) if subset[0] == 'x' else np.array(sel)
    return np.concatenate([G[g] for g in subset.split('+')])


def load(spec, name, root=None):
    parts = []
    for item in spec.split('|'):
        src, subset = item.split(':')
        if root is not None:
            p = root / src / f'{name}.npz'
        else:
            p = V3 / src / f'{name}.npz'
            if not p.exists():
                p = DS / src / f'{name}.npz'
        f = np.load(p)['feats']
        if f.ndim == 2:
            f = f[:, None]
        parts.append(f[..., cols_of(subset, f.shape[-1])])
    return np.concatenate(parts, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--spec', default='stats:all|cnn0:raw|stx_bisp:raw|stx_prof:raw|stx_nois:x7.8.25.26.53')
    ap.add_argument('--tag', default='g1')
    ap.add_argument('--test', action='store_true')
    ap.add_argument('--combine', default='Sd,C')
    ap.add_argument('--steps', type=int, default=3000)
    args = ap.parse_args()
    out = RUNS / args.tag; out.mkdir(parents=True, exist_ok=True)
    dev = 'cuda'
    lab = {n: np.load(V3 / 'stats' / f'{n}.npz')['labels'] for n in ('fit',)}
    mp = out / 'model.pkl'
    if mp.exists():
        lk = pickle.load(open(mp, 'rb'))['lk']
    else:
        lk = Likelihood(dev, 0.02).fit(load(args.spec, 'fit'), lab['fit'], steps=args.steps, width=256,
                                       jitter=0.1, log=lambda m: print(m, flush=True))
        pickle.dump(dict(lk=lk, spec=args.spec), open(mp, 'wb'))

    cdir = out / 'cache'; cdir.mkdir(exist_ok=True)

    def cached(key, fn):
        p = cdir / f'{key}.npz'
        if p.exists():
            z = np.load(p); return z['g'], z['th'], z['e']
        g, th, e = fn()
        np.savez(p, g=g.astype(np.float32), th=th, e=e)
        return g, th, e

    def resid_raw(f):
        f = f.reshape(-1, f.shape[-1])
        gs, ths, es = [], [], []
        for lo in range(0, len(f), 4096):
            e, th = lk.profile(f[lo:lo + 4096])
            x = lk.standardize(f[lo:lo + 4096])
            t = torch.as_tensor((th - lk.t_mean) / lk.t_std, dtype=torch.float32, device=lk.dev)
            with torch.no_grad():
                mu, lv = lk.net(t)
                g = lk.copula((x - mu) * torch.exp(-0.5 * lv))
            gs.append(g.cpu().numpy()); ths.append(th); es.append(lk.last_copula)
        return np.concatenate(gs), np.concatenate(ths), np.concatenate(es)

    Rinv = lk.Rc_inv.cpu().numpy().astype(np.float64)
    gn, _, _ = cached('ts_normal', lambda: resid_raw(load(args.spec, 'normal', TS)))
    names, D = [], []
    for fam, ss in FAMS.items():
        for s in ss:
            if (TS / 'stats' / f'{s}.npz').exists():
                gsft, _, _ = cached('ts_' + s, lambda: resid_raw(load(args.spec, s, TS)))
                D.append((gsft - gn).mean(0)); names.append(s)
    D = np.stack(D, 1)                     # (p, K)
    V = Rinv @ D
    print('directions', names, flush=True)
    gc, thc, ec = cached('calib', lambda: resid_raw(load(args.spec, 'calib')))
    pc = gc @ V
    pm, ps = pc.mean(0), pc.std(0)
    fam_of = {s: f for f, ss in FAMS.items() for s in ss}

    def raw_T(g, cols):
        z = ((g @ V - pm) / ps)[:, cols]
        zc = ((pc - pm) / ps)[:, cols]
        Ci = np.linalg.inv(np.cov(zc.T).reshape(len(cols), len(cols)) + 1e-6 * np.eye(len(cols)))
        t = np.einsum('ni,ij,nj->n', z, Ci, z)
        return np.clip(sst.norm.ppf(np.clip(sst.chi2.cdf(t, len(cols)), 1e-300, 1 - 1e-16)), -8.3, 8.3)

    std_cache = {}

    def T(g, th, cols):
        key = tuple(cols)
        if key not in std_cache:
            std_cache[key] = Standardizer().fit(raw_T(gc, cols), thc, lk.t_mean, lk.t_std)
        return std_cache[key](raw_T(g, cols), th)

    std_omni = Standardizer().fit(ec, thc, lk.t_mean, lk.t_std)
    ga, tha, ea = cached('audit', lambda: resid_raw(load(args.spec, 'audit')))
    allc = list(range(len(names)))
    za_all = T(ga, tha, allc)
    zo_a = std_omni(ea, tha)
    zcomb = {t: dict(np.load(RUNS / 'r2' / f'z_{t}.npz')) for t in filter(None, args.combine.split(','))}
    eval_shifts = sorted({p.stem for p in (V3 / 'stats').glob('shift_*.npz')} |
                         {p.stem for p in (DS / 'stats').glob('shift_*.npz')})
    res, zsave = {}, dict(audit=za_all, calib=T(gc, thc, allc), omni_audit=zo_a)
    for sh in eval_shifts:
        s = sh.removeprefix('shift_')
        fam = fam_of.get(s)
        cols = [i for i, nm in enumerate(names) if fam is None or fam_of[nm] != fam]
        g, th, e = cached(sh, lambda: resid_raw(load(args.spec, sh)))
        zs_all = T(g, th, allc)
        zsave[sh] = zs_all
        r = dict(infam=round(phase2_score(za_all, zs_all), 4), omni=round(phase2_score(zo_a, std_omni(e, th)), 4))
        if cols:
            za, zs = T(ga, tha, cols), T(g, th, cols)
            r['lofo'] = round(phase2_score(za, zs), 4)
            if zcomb and all(sh in zcomb[t] for t in zcomb):
                ba = np.mean([zcomb[t]['audit'] for t in zcomb], 0); bs = np.mean([zcomb[t][sh] for t in zcomb], 0)
                r['combo_only'] = round(phase2_score(ba, bs), 4)
                k = len(zcomb)
                r['lofo+combo'] = round(phase2_score((za + k * ba) / (1 + k), (zs + k * bs) / (1 + k)), 4)
        res[s] = r
        print(f'{s:14s} fam={str(fam):9s} ' + ' '.join(f'{k}={v:.3f}' for k, v in r.items()), flush=True)
    print('normal audit quantiles z_D:', np.quantile(za_all, [.5, .9, .99, .995, .999]).round(3).tolist(),
          'omni:', np.quantile(zo_a, [.5, .9, .99, .995, .999]).round(3).tolist())
    std_shifts = [s for s in res if (V3 / 'stats' / f'shift_{s}.npz').exists()]
    for k in ('lofo', 'lofo+combo', 'combo_only', 'omni', 'infam'):
        vals = [res[s][k] for s in std_shifts if k in res[s]]
        print(f'mean over {len(vals)} standard shifts {k}: {np.mean(vals):.4f}', flush=True)
    if args.test:
        g, th, e = cached('test', lambda: resid_raw(load(args.spec, 'test')))
        zsave['test'] = T(g, th, allc)
        zsave['omni_test'] = std_omni(e, th)
    np.savez(out / 'z_D.npz', **zsave)
    (out / 'report.json').write_text(json.dumps(dict(spec=args.spec, directions=names, shifts=res), indent=1))


if __name__ == '__main__':
    main()
