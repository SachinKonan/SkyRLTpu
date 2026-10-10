"""Feature-factory grader: marginal leave-cosmology-out (LCO) gain of one candidate feature family.

  python factory/grade.py <candidate.py> --out <dir> [--mode screen|full]

The candidate file defines

    def features(maps, mask):   # maps: torch.float32 (B, 1424, 176) on the GPU, zero outside the footprint
        ...                     # mask: torch.bool (1424, 176) on the same device
        return feats            # (B, d) torch tensor or numpy array, finite, a pure per-map function

Pipeline (MOMO_HILLCLIMB_ENV.md section 3):
  0. contract checks (hard fail): per-map (batch vs singles, shuffled order), determinism, finite, shape,
     runtime extrapolated to 10k maps.
  1. features on the frozen noisy dumps (fair-data/agents/diffusion/noisy; never the public test file).
  2. per LCO fold k (20 held-out cosmologies): refit the family on fit3 minus the fold (full mode: the
     conditional likelihood, as holdout3.py; both modes: a ridge theta-probe, as probes.py).
  3. add it to the cached stack (v3-12 recipe without Oh) as a misfit member (weight grid) and/or as an
     extra vote in the consRplus disagreement view; pick one configuration for all folds.
  4. reward = mean(visible fold gains) + min(visible fold gains) - tail penalty; hidden shift families only
     decide hidden_ok and are written to a separate field.
Writes <out>/result.json (everything) and <out>/proposer_view.json (no hidden-shift information).
"""
import argparse
import hashlib
import importlib.util
import json
import pickle
import shutil
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'agents' / 'consistency'))
import data as D  # noqa: E402
import lib  # noqa: E402
from analyze import standardize, mean_of  # noqa: E402
from holdout3 import masks, VARIANTS  # noqa: E402
from jlik import JLikelihood  # noqa: E402
from lco3_analyze import components, cons, pred_sets  # noqa: E402
from model import phase2_score  # noqa: E402
from probes import fold_probe, ridge_fit, RB  # noqa: E402

DUMPS = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion/noisy')
CACHE = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/factory/cache')
FOLDS = (0, 1, 2)
# one hidden shift per physics family (the visible siblings stay); see README.md
HIDDEN = ['baryDMO', 'mult-0.05', 'noise_t5', 'aniso0.7', 'ia0.2', 'peakcut0.3']
VISIBLE = [s for s in lib.SHIFTS if s not in HIDDEN]
NAMES = ['calib', 'in', 'out'] + ['sh_' + s for s in lib.SHIFTS]
# v3-12 without the outlier-exposure head (its LCO gain is inflated by generator overlap)
STACK = {'Sd': .5, 'C': 1, 'Dr': 2.5, 'xSL': 1, 'consP8': .5, 'consRplus': 3, 'N2': .5, 'PsBh': .5, 'SS': 1}
VIEWS0 = RB + ['den_b']
MIS_W = {'screen': [0], 'full': [0, .5, 1, 2]}
VOTES = {'screen': ['none', 'ridge'], 'full': ['none', 'ridge', 'prof']}
TAIL_PENALTY = 0.01          # reward units per percentage point of extra 1% / 0.5% exceedance
CHUNK, BATCH = 512, 64


def log(msg):
    print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ----------------------------------------------------------------------------------------- stack cache
def build_cache(k):
    """Stack members and consRplus views for fold k (CPU, once)."""
    p = CACHE / f'fold{k}.pkl'
    if p.exists():
        with open(p, 'rb') as fh:
            return pickle.load(fh)
    t = time.time()
    comp = components(k, 'std')
    d = lib.OUT / 'lco3' / f'set{k}_std'
    TH = {}
    for f in RB:
        if f.startswith('pred'):
            continue
        with np.load(d / f'{f}.npz') as H:
            TH[f] = {n: H[f'{n}/th'] for n in NAMES}
    TH['pred0'] = pred_sets('cnn0', k)
    TH['pred1'] = pred_sets('cnn1', k)
    TH['den_b'], _, _ = fold_probe('den_b', k)
    keep = {m: comp[m] for m in STACK if m != 'consRplus'}
    C = dict(comp=keep, TH=TH)
    CACHE.mkdir(parents=True, exist_ok=True)
    with open(p.with_suffix('.tmp'), 'wb') as fh:
        pickle.dump(C, fh)
    p.with_suffix('.tmp').replace(p)
    log(f'stack cache fold {k} built in {time.time()-t:.0f}s')
    return C


# ------------------------------------------------------------------------------------------- rows
def rowsets(mode):
    n = {p: len(D.index(p + '3')) for p in ('fit', 'calib', 'audit')}
    held = np.zeros(n['audit'], bool)
    for k in FOLDS:
        held |= masks(k)['audit']
    held = np.nonzero(held)[0]
    if mode == 'full':
        return dict(fit=(np.arange(n['fit']), 2), calib=(np.arange(n['calib']), 2),
                    audit=(np.arange(n['audit']), 1), shift=(held, 1))
    return dict(fit=(np.arange(0, n['fit'], 5), 1), calib=(np.arange(n['calib']), 2),
                audit=(np.arange(n['audit']), 1), shift=(held[::3], 1))


# --------------------------------------------------------------------------------------- candidate
def load_candidate(path):
    spec = importlib.util.spec_from_file_location('candidate', path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    if not callable(getattr(mod, 'features', None)):
        raise ValueError('candidate defines no features(maps, mask)')
    return mod.features


class Runner:
    def __init__(self, fn, dev):
        self.fn, self.dev = fn, dev
        mask = D.mask()
        self.mask_b = torch.as_tensor(mask, device=dev, dtype=torch.bool)
        self.sel = self.mask_b
        self.gpu_s = 0.0
        self.nmaps = 0

    def full(self, rows16):
        x = torch.zeros((len(rows16), 1424, 176), device=self.dev)
        a = np.asarray(rows16)
        x[:, self.sel] = torch.as_tensor(a if a.flags.writeable else a.copy(), device=self.dev).float()
        return x

    @torch.no_grad()
    def call(self, x):
        f = self.fn(x, self.mask_b)
        if isinstance(f, torch.Tensor):
            f = f.detach().float().cpu().numpy()
        f = np.asarray(f, dtype=np.float32)
        if f.ndim != 2 or f.shape[0] != x.shape[0]:
            raise ValueError(f'features must return (B, d); got {f.shape} for B={x.shape[0]}')
        return f

    def run_rows(self, arr, rows):
        """Features of arr[rows] (memmap of valid-pixel vectors), chunked I/O with one-chunk prefetch."""
        out = []
        chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
        with ThreadPoolExecutor(1) as ex:
            fut = ex.submit(lambda r: np.asarray(arr[r]), chunks[0])
            for j in range(len(chunks)):
                block = fut.result()
                if j + 1 < len(chunks):
                    fut = ex.submit(lambda r: np.asarray(arr[r]), chunks[j + 1])
                if self.dev.type == 'cuda':
                    torch.cuda.synchronize()
                t = time.time()
                for b in range(0, len(block), BATCH):
                    out.append(self.call(self.full(block[b:b + BATCH])))
                if self.dev.type == 'cuda':
                    torch.cuda.synchronize()
                self.gpu_s += time.time() - t
                self.nmaps += len(block)
        return np.concatenate(out)


def contract(R, budget):
    """Hard checks on 8 audit maps + runtime on 512 audit maps. Returns (ok, info)."""
    arr = np.load(DUMPS / 'audit_v0.npy', mmap_mode='r')
    x = R.full(arr[:8])
    info = {}
    try:
        a = R.call(x)
        info['dim'] = int(a.shape[1])
        if not 1 <= a.shape[1] <= 4096:
            return False, dict(info, fail=f'feature dimension {a.shape[1]} outside [1, 4096]')
        if not np.isfinite(a).all():
            return False, dict(info, fail='non-finite features on normal maps')
        scale = np.abs(a).max(0) + 1e-6
        single = np.concatenate([R.call(x[i:i + 1]) for i in range(8)])
        perm = np.array([5, 2, 7, 0, 3, 6, 1, 4])
        shuf = R.call(x[torch.as_tensor(perm, device=R.dev)])[np.argsort(perm)]
        again = R.call(x)
        # entries off by more than 1e-3 of the column scale; a few are tolerated because GPU kernels may
        # round differently per batch size (or use float atomics) and flip a value across a histogram edge
        for key, other, tol in (('per_map', single, 1e-3), ('shuffle', shuf, 1e-3), ('determinism', again, 1e-3)):
            rel = np.abs(other - a) / scale
            info[f'{key}_max_rel'] = float(rel.max())
            info[f'{key}_frac_off'] = float((rel > tol).mean())
        if info['per_map_frac_off'] > 0.01 or info['shuffle_frac_off'] > 0.01:
            return False, dict(info, fail='not a per-map function (batch-of-1 or shuffled order differs)')
        if info['determinism_frac_off'] > 0.01:
            return False, dict(info, fail='not deterministic')
        g0, n0 = R.gpu_s, R.nmaps
        R.run_rows(arr, np.arange(64))                       # warm-up
        R.gpu_s, R.nmaps = g0, n0
        t = time.time()
        R.run_rows(arr, np.arange(64, 576))
        sec10k = (time.time() - t) / 512 * 10000
        R.gpu_s, R.nmaps = g0, n0
        info['sec_per_10k'] = round(sec10k, 2)
        info['budget_sec_per_10k'] = budget
        if sec10k > budget:
            return False, dict(info, fail=f'runtime {sec10k:.0f}s per 10k maps exceeds budget {budget}s')
    except Exception as e:  # noqa: BLE001 - any candidate error is a contract failure
        return False, dict(info, fail=f'{type(e).__name__}: {e}')
    return True, info


def extract(R, rows, fdir):
    """Feature arrays (n, V, d) for fit / calib / audit / every shift; cached as .npy per set."""
    fdir.mkdir(parents=True, exist_ok=True)
    F = {}
    jobs = [('fit',) + rows['fit'], ('calib',) + rows['calib'], ('audit',) + rows['audit']]
    jobs += [('shift_' + s,) + rows['shift'] for s in lib.SHIFTS]
    for name, idx, views in jobs:
        p = fdir / f'{name}.npy'
        if p.exists():
            F[name] = np.load(p)
            continue
        t = time.time()
        parts = [R.run_rows(np.load(DUMPS / f'{name}_v{v}.npy', mmap_mode='r'), idx) for v in range(views)]
        f = np.stack(parts, 1)
        if not np.isfinite(f).all():
            raise FloatingPointError(f'non-finite features in {name}')
        np.save(fdir / f'{name}.tmp.npy', f)
        (fdir / f'{name}.tmp.npy').replace(p)
        F[name] = f
        log(f'  features {name} {f.shape} {time.time()-t:.0f}s')
    return F


# ------------------------------------------------------------------------------------------ folds
def fold_inputs(F, rows, k):
    m = masks(k)
    fitrows = rows['fit'][0]
    mf = m['fit'][fitrows]
    Lfit_all = lib.labels('fit')
    tm, ts = Lfit_all[~m['fit']].mean(0), Lfit_all[~m['fit']].std(0)   # the stack's normalization
    d = F['fit'].shape[-1]
    S = dict(fit=F['fit'][~mf], calib=F['calib'][~m['calib']], out=np.concatenate(
        [F['calib'][m['calib']].reshape(-1, 1, d), F['audit'][m['audit']]], 0))
    S['in'] = F['audit'][~m['audit']]
    shrows = rows['shift'][0]
    held = np.nonzero(m['audit'])[0]
    take = np.isin(held, shrows)
    pos = np.searchsorted(shrows, held[take])
    for s in lib.SHIFTS:
        S['sh_' + s] = F['shift_' + s][pos]
    Lcal = lib.labels('calib')[~m['calib']]
    return S, Lfit_all[fitrows][~mf], Lcal, tm, ts, take


def ridge_vote(S, Lfit, Lcal, tm, ts):
    d = S['fit'].shape[-1]
    Xf = S['fit'].reshape(-1, d).astype(np.float64)
    yf = np.repeat((Lfit - tm) / ts, S['fit'].shape[1], 0)
    Xc = S['calib'].reshape(-1, d).astype(np.float64)
    yc = np.repeat((Lcal - tm) / ts, S['calib'].shape[1], 0)
    f, a, mse = ridge_fit(Xf, yf, Xc, yc)
    per = ((f(Xc) - yc) ** 2).mean(0)
    return {n: f(S[n].reshape(-1, d).astype(np.float64)).astype(np.float32) for n in NAMES}, \
        dict(alpha=a, calib_mse=round(float(mse), 4), calib_mse_per_param=np.round(per, 4).tolist())


def lik_member(S, Lfit, dev):
    lk = JLikelihood(dev, 0.02, VARIANTS['std']['jm']).fit(S['fit'], Lfit, steps=VARIANTS['std']['steps'],
                                                           width=VARIANTS['std']['width'], log=lambda m: None,
                                                           jitter=0.1)
    cop, th = {}, {}
    for n in NAMES:
        x = S[n]
        E, t = lk.profile(x.reshape(-1, x.shape[-1]))
        cop[n] = lk.last_copula
        th[n] = ((t - lk.t_mean) / lk.t_std).astype(np.float32)
    return cop, th


def sub(dct, take):
    return {n: (v[take] if n.startswith('sh_') else v) for n, v in dct.items()}


def metrics(z):
    sh = {s: phase2_score(z['out'], z['sh_' + s]) for s in lib.SHIFTS}
    t = np.quantile(z['in'], [.99, .995, .999])
    exc = [float(np.mean(z['out'] > x)) * 100 for x in t]
    return sh, exc


# ------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidate', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--mode', default='screen', choices=['screen', 'full'])
    ap.add_argument('--budget', type=float, default=60.0, help='max seconds per 10k maps (hard fail)')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()
    t0 = time.time()
    dev = torch.device(args.device)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    src = args.candidate.read_text()
    sha = hashlib.sha256(src.encode()).hexdigest()[:16]
    meta = out / 'meta.json'
    if meta.exists() and json.loads(meta.read_text())['sha'] != sha:
        log('candidate changed since the cached run: clearing cached features and folds')
        for p in out.iterdir():
            shutil.rmtree(p) if p.is_dir() else p.unlink()
    meta.write_text(json.dumps(dict(candidate=str(args.candidate.resolve()), sha=sha)))
    shutil.copy(args.candidate, out / 'candidate.py')
    res = dict(candidate=str(args.candidate.resolve()), sha=sha, mode=args.mode, ok=False, reward=None)

    def finish(r):
        (out / 'result.json').write_text(json.dumps(r, indent=1))
        pv = {k: v for k, v in r.items() if k != 'hidden'}
        pv['hidden_ok'] = r.get('hidden', {}).get('hidden_ok')
        (out / 'proposer_view.json').write_text(json.dumps(pv, indent=1))
        log(f'result: ok={r["ok"]} reward={r["reward"]} hidden_ok={pv["hidden_ok"]} -> {out / "result.json"}')

    torch.manual_seed(0)
    try:
        fn = load_candidate(args.candidate)
    except Exception as e:  # noqa: BLE001
        res['contract'] = dict(fail=f'import: {type(e).__name__}: {e}')
        return finish(res)
    R = Runner(fn, dev)
    ok, info = contract(R, args.budget)
    res['contract'] = info
    log(f'contract ok={ok} {info}')
    if not ok:
        return finish(res)

    rows = rowsets(args.mode)
    caches = {k: build_cache(k) for k in FOLDS}
    t1 = time.time()
    try:
        F = extract(R, rows, out / f'feats_{args.mode}')
    except Exception as e:  # noqa: BLE001
        res['contract']['fail'] = f'extraction: {type(e).__name__}: {e}'
        return finish(res)
    t_extract = time.time() - t1
    log(f'features done: {R.nmaps} maps computed this run, {R.gpu_s:.0f}s in features(), {t_extract:.0f}s wall')

    # per-fold candidate parts (cached)
    t2 = time.time()
    parts, ridge_info = {}, {}
    for k in FOLDS:
        p = out / f'fold{k}_{args.mode}.pkl'
        if p.exists():
            with open(p, 'rb') as fh:
                parts[k] = pickle.load(fh)
        else:
            S, Lfit, Lcal, tm, ts, take = fold_inputs(F, rows, k)
            P = dict(take=take)
            P['ridge'], P['ridge_info'] = ridge_vote(S, Lfit, Lcal, tm, ts)
            if args.mode == 'full':
                cop, th = lik_member(S, Lfit, args.device)
                P['prof'] = th
                P['mis'] = standardize(cop, th)
            with open(p.with_suffix('.tmp'), 'wb') as fh:
                pickle.dump(P, fh)
            p.with_suffix('.tmp').replace(p)
            parts[k] = P
            log(f'  fold {k} fitted')
        ridge_info[k] = parts[k]['ridge_info']
    t_fit = time.time() - t2

    # configurations: misfit weight x vote; one configuration for all folds
    configs = [(w, v) for w in MIS_W[args.mode] for v in VOTES[args.mode] if (w, v) != (0, 'none')]
    base, cand = {}, {c: {} for c in configs}
    standalone = {}
    for k in FOLDS:
        C, P = caches[k], parts[k]
        take = P['take']
        comp = {m: sub(z, take) for m, z in C['comp'].items()}
        TH = {f: sub(v, take) for f, v in C['TH'].items()}
        comp['consRplus'] = cons(TH, VIEWS0, NAMES)
        base[k] = metrics(mean_of(*[comp[m] for m in STACK], w=list(STACK.values())))
        consv = {}
        for v in VOTES[args.mode]:
            if v != 'none':
                TH['cand'] = P[v]
                consv[v] = cons(TH, VIEWS0 + ['cand'], NAMES)
        if 'mis' in P:
            standalone.setdefault('misfit', []).append(metrics(P['mis'])[0])
        for v in consv:
            standalone.setdefault(f'consRplus+{v}', []).append(metrics(consv[v])[0])
        standalone.setdefault('consRplus', []).append(metrics(comp['consRplus'])[0])
        for (w, v) in configs:
            c = dict(comp)
            W = dict(STACK)
            if v != 'none':
                c['consRplus'] = consv[v]
            if w > 0:
                c['X'] = P['mis']; W['X'] = w
            cand[(w, v)][k] = metrics(mean_of(*[c[m] for m in W], w=list(W.values())))

    def agg(sh, which):
        return float(np.mean([sh[s] for s in which]))

    def summarize(cfg):
        vis = [agg(cand[cfg][k][0], VISIBLE) - agg(base[k][0], VISIBLE) for k in FOLDS]
        hid = [agg(cand[cfg][k][0], HIDDEN) - agg(base[k][0], HIDDEN) for k in FOLDS]
        d_exc = np.mean([np.array(cand[cfg][k][1]) - np.array(base[k][1]) for k in FOLDS], 0)
        pen = TAIL_PENALTY * (max(0., d_exc[0]) + max(0., d_exc[1]))
        return dict(vis=vis, hid=hid, d_exc=d_exc.tolist(), pen=pen, reward=float(np.mean(vis) + min(vis) - pen))

    S_ = {cfg: summarize(cfg) for cfg in configs}
    best = max(configs, key=lambda c: S_[c]['reward'])
    b = S_[best]
    per_shift = {s: round(float(np.mean([cand[best][k][0][s] - base[k][0][s] for k in FOLDS])), 4)
                 for s in lib.SHIFTS}
    exc_b = np.mean([base[k][1] for k in FOLDS], 0)
    exc_c = np.mean([cand[best][k][1] for k in FOLDS], 0)
    sa = {name: round(float(np.mean([agg(x, VISIBLE) for x in v])), 4) for name, v in standalone.items()}
    res.update(
        ok=True, reward=round(b['reward'], 5),
        config=dict(misfit_weight=best[0], vote=best[1]),
        visible=dict(
            shifts=VISIBLE,
            per_fold_gain=[round(x, 4) for x in b['vis']], mean_gain=round(float(np.mean(b['vis'])), 4),
            base_mean=round(float(np.mean([agg(base[k][0], VISIBLE) for k in FOLDS])), 4),
            cand_mean=round(float(np.mean([agg(cand[best][k][0], VISIBLE) for k in FOLDS])), 4),
            per_shift_gain={s: per_shift[s] for s in VISIBLE},
            standalone_visible_mean=sa,
            configs=[dict(misfit_weight=c[0], vote=c[1], reward=round(S_[c]['reward'], 5),
                          per_fold_gain=[round(x, 4) for x in S_[c]['vis']]) for c in configs],
            theta_probe=ridge_info),
        tail=dict(exc_base_pct=np.round(exc_b, 3).tolist(), exc_cand_pct=np.round(exc_c, 3).tolist(),
                  nominal_pct=[1, .5, .1], penalty=round(b['pen'], 5)),
        runtime=dict(sec_per_1k_maps=round(R.gpu_s / max(R.nmaps, 1) * 1000, 3) if R.nmaps else None,
                     maps_this_run=R.nmaps, extract_s=round(t_extract, 1), fit_s=round(t_fit, 1),
                     total_s=round(time.time() - t0, 1)),
        hidden=dict(shifts=HIDDEN, hidden_ok=bool(np.mean(b['hid']) > 0),
                    per_fold_gain=[round(x, 4) for x in b['hid']], mean_gain=round(float(np.mean(b['hid'])), 4),
                    per_shift_gain={s: per_shift[s] for s in HIDDEN},
                    all19_base=round(float(np.mean([agg(base[k][0], lib.SHIFTS) for k in FOLDS])), 4),
                    all19_cand=round(float(np.mean([agg(cand[best][k][0], lib.SHIFTS) for k in FOLDS])), 4)))
    finish(res)


if __name__ == '__main__':
    main()
