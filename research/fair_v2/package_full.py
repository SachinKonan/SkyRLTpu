"""Package any export3-style recipe (consistency agent) as a Codabench Phase-2 code submission.

  python package_full.py --name v3-04 --recipe '{"Sd":1,"C":1,"Dr":1,"xSL":1,"consP8":1}' [--include Sd,C,...]
  python package_full.py --name BIG_RE2 --recipe-file .../export3/recipe_BIG_RE2.txt

Writes /scratch/gpfs/ZHUANGL/sk7524/fair-data/packages/<name>/ and <name>.zip with model.py at the root.
Members (components) and their frozen artifacts mirror agents/consistency/export3.py exactly:
  likelihood members  Sd=dog C=cnn C1=cnn1 Dr=den_r N2=N2 K=stx_pkcl PsBh=PsBh X=dirpower (fam/model_<family>.pkl)
  external likelihood members (v3-12/v3-13):
    SS  diffusion small-scale family on difa (agents/diffusion2/nlevel.py, z as nlevel_eval.py)
    Oh  outlier-exposure head family on oe_head (fit_v3 run agents/outlier2/runs/r1, its frozen Standardizer)
  consistency members consP8, consR (calib3 Mahalanobis of family-estimate deviations), xSL (cross-scale),
    consRplus12 = consR views + den_b ridge theta-probe (v3-12), consRplus = + dirpower probe (v3-13).
Each component's Standardizer is refit here from the stored calib3 arrays with the same code path as
export3 (deterministic), and a CPU parity check recomputes every component's frozen test z from the stored
per-map family outputs using the packaged inference math (fwl_lik.component_z).
`--include` adds components with weight 0 (edit config.json weights later without repackaging).
CPU only; needs the fair-env python (no GPU).
"""
import argparse
import io
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
CONS_DIR = HERE / 'agents' / 'consistency'
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(CONS_DIR))

# family pickles hold CUDA tensors from della-gh: load them on CPU
torch.storage._load_from_bytes = lambda b: torch.load(io.BytesIO(b), map_location='cpu', weights_only=False)

import lib  # noqa: E402  (consistency agent helpers; read-only)
from analyze import TM, TS, raw_theta  # noqa: E402
from model import Standardizer  # noqa: E402
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('fwl_lik', HERE / 'pack_full' / 'fwl_lik.py')
_fwl = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(_fwl)
component_z = _fwl.component_z

DATA = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data')
PKG = DATA / 'packages'
MEMBERS = {'Sd': 'dog', 'C': 'cnn', 'C1': 'cnn1', 'Dr': 'den_r', 'N2': 'N2', 'K': 'stx_pkcl', 'PsBh': 'PsBh',
           'X': 'dirpower'}
RB = ['b0', 'b1', 'b2', 'b3', 'cnn', 'cnn1', 'pred0', 'pred1', 'dog', 'den_r', 'den_sd']
CONS = {'consP8': ['psL', 'psS', 'onept', 'extrema', 'mf', 'dog', 'cnn', 'pred0'],
        'consR': RB, 'consRplus12': RB + ['pden_b'], 'consRplus': RB + ['pden_b', 'pdirpower']}
# ridge theta-probe views -> (feature source, fitting recipe): 'cv' = consistency2/c2probe.full_probe (v3-12's
# den_b view), 'calib' = consistency/probes.full_probe (factory/export_family.py's family view)
RIDGE = {'pden_b': 'den_b', 'pdirpower': 'dirpower'}
RIDGE_METHOD = {'pden_b': 'cv', 'pdirpower': 'calib'}
PDIR = DATA / 'agents' / 'consistency2' / 'probes'
NLFAM = DATA / 'agents' / 'diffusion2' / 'nlevel' / 'fam'
OH_RUN = DATA / 'agents' / 'outlier2' / 'runs' / 'r1'
OE_CKPT = DATA / 'agents' / 'outlier' / 'run_all14' / 'ckpt.pt'     # reproduces v3/oe bit-exactly on GH200
DUNET = DATA / 'agents' / 'diffusion' / 'feats_r1' / 'model.pt'     # frozen model of the difa source
EXT = ('SS', 'Oh')
ALIASES = {'consRbands': 'consR'}
CNN_CKPT = {'cnn0': DATA / 'v3' / 'cnn_b' / 'net.pt', 'cnn1': DATA / 'v3' / 'cnn_a' / 'net.pt'}
UNET = DATA / 'agents' / 'denoiser' / 'unet_final.pt'
EDGES = {'edges_stats': DATA / 'feats-v2' / 'edges.npz',
         'edges_denoised': DATA / 'v3' / 'den' / 'edges_denoised.npz',
         'edges_nois': DATA / 'agents' / 'statistics' / 'edges_nois.npz',
         'edges_fam': DATA / 'agents' / 'statistics' / 'edges_fam.npz'}
OFFSETS = PKG / '_cache' / 'profile_offsets_gh200.npy'   # CUDA torch.Generator(seed 0) restart offsets
SNAPSHOTS = {'fwl_stats.py': HERE / 'stats.py', 'fwl_cnn.py': HERE / 'pack' / 'cnn_infer.py',
             'fwl_unet.py': HERE / 'agents' / 'denoiser' / 'unet.py',
             'fwl_famfeats.py': HERE / 'agents' / 'statistics' / 'famfeats.py',
             'fwl_dunet.py': HERE / 'agents' / 'diffusion' / 'unet.py',
             'fwl_dirpower.py': HERE / 'factory' / 'examples' / 'dirpower.py'}


def T(a, dt=torch.float64):
    return torch.as_tensor(np.asarray(a)).to(dt).clone()


def compact(sd):
    """Conv / linear weights (>= 2-D) stored as bf16: lossless under the bf16 autocast used at inference, which
    casts them to bf16 anyway; BatchNorm parameters/statistics and biases stay fp32. Halves the zip size."""
    return {k: (v.cpu().to(torch.bfloat16) if v.is_floating_point() and v.dim() >= 2 else v.cpu())
            for k, v in sd.items()}


def fam_dir(var):
    return lib.OUT / (f'fam_{var}' if var else 'fam')


def fam_npz(f, var=None):
    with np.load(fam_dir(var) / f'{f}.npz') as d:
        return {k: d[k] for k in d.files}


def family_sources(f):
    spec = lib.FAMILIES[f]
    spec = spec if isinstance(spec, list) else [spec]
    return [[src, [int(i) for i in np.asarray(c)]] for src, c in spec]


def export_family(f, var, offsets):
    with open(fam_dir(var) / f'model_{f}.pkl', 'rb') as fh:
        lk = pickle.load(fh)
    return export_lk(lk, f, var, family_sources(f), offsets)


def ss_cols():
    sys.path.insert(0, str(HERE / 'agents' / 'diffusion2'))
    from nlevel import fam_cols
    return [int(i) for i in fam_cols()['SS']]


def ext_member(c):
    """(Likelihood, sources, frozen std dict) of an external likelihood member."""
    if c == 'SS':
        lk = torch.load(NLFAM / 'model_SS.pt', map_location='cpu', weights_only=False)
        with np.load(NLFAM / 'SS.npz') as d:      # theta stored RAW (model.Likelihood.profile)
            s = Standardizer().fit(d['calib/cop'], d['calib/th'], TM, TS)
        return lk, [['difa', ss_cols()]], dict(loc=T(s.loc), scale=T(s.scale), norm=float(s.norm), t_mean=T(TM),
                                                t_std=T(TS))
    with open(OH_RUN / 'model_Oh.pkl', 'rb') as fh:
        d = pickle.load(fh)
    assert d['spec'] == 'oe_head:raw' and d['stat'] == 'copula', d['spec']
    lk, s = d['lk'], d['std']
    d_oe = int(np.asarray(lk.f_mean).shape[-1])
    return lk, [['oe_head', list(range(d_oe))]], dict(loc=T(s.loc), scale=T(s.scale), norm=float(s.norm),
                                                      t_mean=T(lk.t_mean), t_std=T(lk.t_std))


def ridge_params_cv(src, nfold=5, seed=0):
    """consistency2/c2probe.full_probe + ridge_cv (cosmology-grouped CV on fit3), returning the map's parameters."""
    L = lib.labels('fit')
    tm, ts = L.mean(0), L.std(0)
    Xf = lib.load(src, 'fit')
    V = Xf.shape[1]
    y = np.repeat((L - tm) / ts, V, 0)
    groups = np.repeat(lib.load('stats', 'fit', 'index')[:, 0], V)
    X = Xf.reshape(-1, Xf.shape[-1]).astype(np.float64)
    mu, sd = X.mean(0), X.std(0) + 1e-8
    A = (X - mu) / sd
    ym = y.mean(0)
    Y = y - ym
    ug = np.unique(groups)
    perm = np.random.default_rng(seed).permutation(ug)
    fold_of = {g: i % nfold for i, g in enumerate(perm)}
    fid = np.array([fold_of[g] for g in groups])
    Gs = [A[fid == j].T @ A[fid == j] for j in range(nfold)]
    bs = [A[fid == j].T @ Y[fid == j] for j in range(nfold)]
    Gt, bt = sum(Gs), sum(bs)
    N = len(A)
    I = np.eye(A.shape[1])
    alphas = np.logspace(-6, 1, 8)
    mses = []
    for a in alphas:
        err = 0.0
        for j in range(nfold):
            n_tr = N - (fid == j).sum()
            W = np.linalg.solve(Gt - Gs[j] + a * n_tr * I, bt - bs[j])
            err += ((A[fid == j] @ W - Y[fid == j]) ** 2).sum()
        mses.append(err / Y.size)
    i = int(np.argmin(mses))
    W = np.linalg.solve(Gt + alphas[i] * N * I, bt)
    return dict(src=src, mu=T(mu), sd=T(sd), W=T(W), ym=T(ym), alpha=float(alphas[i]), mse=float(mses[i]))


def ridge_params(src, method='calib'):
    """probes.full_probe / ridge_fit for one source, returning the fitted map's parameters."""
    if method == 'cv':
        return ridge_params_cv(src)
    Lf, Lc = lib.labels('fit'), lib.labels('calib')
    tm, ts = Lf.mean(0), Lf.std(0)
    Xf, Xc = lib.load(src, 'fit'), lib.load(src, 'calib')
    yf = np.repeat((Lf - tm) / ts, Xf.shape[1], 0)
    yc = np.repeat((Lc - tm) / ts, Xc.shape[1], 0)
    Xf = Xf.reshape(-1, Xf.shape[-1]).astype(np.float64)
    Xc = Xc.reshape(-1, Xc.shape[-1]).astype(np.float64)
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
    A = (Xf - mu) / sd
    ym = yf.mean(0)
    G = A.T @ A
    b = A.T @ (yf - ym)
    best = None
    for a in [1e-1, 1, 10, 100, 1e3, 1e4]:
        W = np.linalg.solve(G + a * len(A) / 1000 * np.eye(len(G)), b)
        mse = np.mean((((Xc - mu) / sd) @ W + ym - yc) ** 2)
        if best is None or mse < best[0]:
            best = (mse, a, W)
    return dict(src=src, mu=T(mu), sd=T(sd), W=T(best[2]), ym=T(ym), alpha=float(best[1]), mse=float(best[0]))


def ridge_np(r):
    return {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in r.items()}


def export_lk(lk, f, var, sources, offsets):
    assert np.allclose(lk.t_mean, TM) and np.allclose(lk.t_std, TS), f'{f}: theta normalization differs'
    cpu = lambda t: t.detach().cpu().clone()
    net_sd = {k: cpu(v) for k, v in lk.net.state_dict().items()}
    reg_sd = {k: cpu(v) for k, v in lk.reg.state_dict().items()}
    width = net_sd['mu.weight'].shape[1]
    depth = sum(1 for k in net_sd if k.startswith('body.') and k.endswith('.weight'))
    return dict(family=f, variant=var or '', sources=sources,
                f_mean=T(lk.f_mean), f_std=T(lk.f_std), keep=torch.as_tensor(np.asarray(lk.keep, dtype=bool)),
                t_mean=T(lk.t_mean), t_std=T(lk.t_std), t_lo=T(lk.t_lo), t_hi=T(lk.t_hi),
                cosmo=cpu(lk.cosmo).float(), hull_r=float(lk.hull_r), offsets=T(offsets, torch.float32),
                width=int(width), depth=int(depth), dk=int(np.asarray(lk.keep).sum()),
                reg_width=int(reg_sd['f.0.weight'].shape[0]), net=net_sd, reg=reg_sd,
                Rinv=cpu(lk.Rinv).float(), Rc_inv=cpu(lk.Rc_inv).float(), cq=cpu(lk.cq).float(),
                cg=cpu(lk.cg).float())


def fit_std(raw_calib, th_calib):
    """analyze.standardize: Standardizer fit on calib3 (raw score vs raw theta)."""
    s = Standardizer().fit(raw_calib, raw_theta(th_calib), TM, TS)
    return dict(loc=T(s.loc), scale=T(s.scale), norm=float(s.norm), t_mean=T(TM), t_std=T(TS))


def preds(name, s):
    return lib.load(name, s)[..., -5:].reshape(-1, 5)


def build_component(c, var):
    """Returns (component params, test-side inputs for the CPU parity check)."""
    if c in EXT:
        lk, sources, std = ext_member(c)
        EXT_BLOBS[c] = (lk, sources)
        return dict(kind='lik', family=c, families=[c], variant='', std=std), {c: 'ext'}
    if c in MEMBERS:
        f = MEMBERS[c]
        v = var.get(c)
        F = fam_npz(f, v)
        p = dict(kind='lik', family=f, families=[f], variant=v or '', std=fit_std(F['calib/cop'], F['calib/th']))
        return p, {f: v}
    if c in CONS:
        ins = CONS[c]
        fams = [f for f in ins if f not in ('pred0', 'pred1') and f not in RIDGE]
        ridge = {}
        for v in ins:
            if v in RIDGE:
                if v not in RIDGE_CACHE:
                    RIDGE_CACHE[v] = ridge_params(RIDGE[v], RIDGE_METHOD[v])
                ridge[v] = RIDGE_CACHE[v]

        def view(f, s):
            if f in ('pred0', 'pred1'):
                return preds('cnn0' if f == 'pred0' else 'cnn1', s)
            if f in ridge:
                X = lib.load(RIDGE[f], s)
                return _fwl.ridge_view(ridge_np(ridge[f]), X.reshape(-1, X.shape[-1]))
            return fam_npz(f)[f'{s}/th']
        A = {}
        for s in ('calib',):
            A[s] = np.stack([view(f, s) for f in ins], 1)
        vec = (A['calib'] - A['calib'].mean(1, keepdims=True))[:, 1:].reshape(len(A['calib']), -1)
        # analyze.mahal (ridge 1e-3, covariance from calib3)
        mu = vec.mean(0)
        C = np.cov(vec - mu, rowvar=False)
        C = C + 1e-3 * np.trace(C) / len(C) * np.eye(len(C))
        Ci = np.linalg.inv(C)
        m2 = np.einsum('ij,jk,ik->i', vec - mu, Ci, vec - mu)
        p = dict(kind='cons', inputs=ins, families=fams, preds=[f for f in ins if f not in fams],
                 mu=torch.as_tensor(mu.copy()), Ci=T(Ci), std=fit_std(np.log(m2 + 1.0), A['calib'].mean(1)))
        if ridge:
            p['ridge'] = ridge
        return p, {f: None for f in fams}
    if c == 'xSL':
        with np.load(lib.OUT / 'scale' / 'scale2.npz') as d:
            X = {k: d[k] for k in ('calib/cop_small@large', 'calib/cop_small@small', 'calib/cop_large@small',
                                   'calib/cop_large@large', 'calib/th_small', 'calib/th_large')}
        raw = (X['calib/cop_small@large'] - X['calib/cop_small@small']) + \
              (X['calib/cop_large@small'] - X['calib/cop_large@large'])
        p = dict(kind='xsl', families=['small', 'large'],
                 std=fit_std(raw, 0.5 * (X['calib/th_small'] + X['calib/th_large'])))
        return p, {'small': None, 'large': None}
    raise SystemExit(f'unknown component {c}')


EXT_BLOBS, RIDGE_CACHE = {}, {}


def np_params(p):
    q = dict(p)
    q['std'] = {k: (v.numpy() if torch.is_tensor(v) else v) for k, v in p['std'].items()}
    for k in ('mu', 'Ci'):
        if k in q:
            q[k] = q[k].numpy()
    if 'ridge' in q:
        q['ridge'] = {v: ridge_np(r) for v, r in q['ridge'].items()}
    return q


def parity_cpu(comps, weights, fam_var, tag, ref_file=''):
    """Recompute frozen test z of every component from stored per-map family outputs (no GPU)."""
    th, cop = {}, {}
    for f, v in fam_var.items():
        if v == 'ext':
            if f == 'SS':
                with np.load(NLFAM / 'SS.npz') as d:
                    th[f], cop[f] = ((d['test/th'] - TM) / TS).astype(np.float32), d['test/cop']
            continue                                   # Oh: no stored per-map test outputs (GPU parity only)
        F = fam_npz(f, v)
        th[f], cop[f] = F['test/th'], F['test/cop']
    pr = {'pred0': preds('cnn0', 'test'), 'pred1': preds('cnn1', 'test')}
    for v, r in RIDGE_CACHE.items():
        X = lib.load(r['src'], 'test')
        pr[v] = _fwl.ridge_view(ridge_np(r), X.reshape(-1, X.shape[-1]))
    if (PDIR / 'den_b_full.npz').exists() and 'pden_b' in pr:
        with np.load(PDIR / 'den_b_full.npz') as d:
            print(f"  ridge pden_b vs stored probe: calib max abs {np.abs(_ridge_calib('pden_b') - d['calib']).max():.2e}"
                  f" test max abs {np.abs(pr['pden_b'] - d['test']).max():.2e}")
    cross = None
    if 'small' in th:
        with np.load(lib.OUT / 'scale' / 'scale2.npz') as d:
            cross = {'small@large': d['test/cop_small@large'], 'large@small': d['test/cop_large@small']}
            for f in ('small', 'large'):
                dd = np.abs(d[f'test/cop_{f}@{f}'] - cop[f]).max()
                print(f'  scale2 cop_{f}@{f} vs fam cop: max abs diff {dd:.2e}')
    ref = {}
    if ref_file:
        with np.load(ref_file) as d:
            ref = {REF_NAMES.get(k, k): d[k] for k in d.files if '/' not in k}
    rp = lib.OUT / 'export3' / f'components_{tag}.npz'
    for p in [rp] + sorted((lib.OUT / 'export3').glob('components_*.npz')):
        if p.exists():
            with np.load(p) as d:
                for k in d.files:
                    c, s = k.split('/', 1)
                    if s == 'test' and c not in ref:
                        ref[c] = d[k]
    z = {}
    for c, p in comps.items():
        if any(f not in th for f in p['families']):
            print(f'  component {c:11s}: needs GPU parity (no stored per-map outputs)')
            z[c] = ref.get(c)
            continue
        z[c] = component_z(c, np_params(p), th, cop, pr, cross)
        if c in ref:
            print(f'  component {c:7s} test z vs export3: max abs diff {np.abs(z[c] - ref[c]).max():.2e}')
        else:
            print(f'  component {c:7s}: no export3 reference')
    W = {c: w for c, w in weights.items() if w}
    tot = sum(w * z[c] for c, w in W.items()) / sum(W.values())
    return tot


REF_NAMES = {'consRplus': 'consRplus12', 'consRplus_x': 'consRplus'}   # export_family names -> package names


def _ridge_calib(v):
    r = RIDGE_CACHE[v]
    X = lib.load(r['src'], 'calib')
    return _fwl.ridge_view(ridge_np(r), X.reshape(-1, X.shape[-1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', required=True)
    ap.add_argument('--recipe', default='', help='JSON weights, e.g. {"Sd":1,"C":1}')
    ap.add_argument('--recipe-file', default='', help='export3 recipe_<tag>.txt')
    ap.add_argument('--include', default='', help='extra components packaged with weight 0')
    ap.add_argument('--ref', default='', help='frozen test z (.npy) to compare the CPU recomposition against')
    ap.add_argument('--ref-components', default='', help='npz of frozen test z per component (export_family names)')
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--profile-chunk', type=int, default=16384,
                    help='fixed row count of each likelihood-profile chunk (zero padded); launch-bound, so large')
    ap.add_argument('--force', action='store_true')
    args = ap.parse_args()
    if args.recipe_file:
        js = json.loads(Path(args.recipe_file).read_text())
        weights, var = js['weights'], js.get('variants', {})
    else:
        weights, var = json.loads(args.recipe), {}
    weights = {ALIASES.get(k, k): float(v) for k, v in weights.items()}
    for c in filter(None, args.include.split(',')):
        weights.setdefault(ALIASES.get(c, c), 0.0)
    out = PKG / args.name
    if out.exists():
        if not args.force:
            raise SystemExit(f'{out} exists (use --force)')
        shutil.rmtree(out)
    out.mkdir(parents=True)
    for dst, src in SNAPSHOTS.items():
        a, b = (HERE / 'pack_full' / dst).read_text(), src.read_text()
        if a != b:
            print(f'WARNING: pack_full/{dst} differs from its source {src} (snapshot kept)')
    for fn in ('model.py', 'fwl_lik.py', 'fwl_den.py', 'fwl_oe.py', 'fwl_dif.py', *SNAPSHOTS):
        shutil.copy(HERE / 'pack_full' / fn, out / fn)
    shutil.copy(DATA / 'WIDE12H_bin2_2arcmin_mask.npy', out / 'WIDE12H_bin2_2arcmin_mask.npy')

    comps, fam_var = {}, {}
    for c in weights:
        p, fv = build_component(c, var)
        comps[c] = p
        for f, v in fv.items():
            if f in fam_var and fam_var[f] != v:
                raise SystemExit(f'family {f} needed with two variants')
            fam_var[f] = v
    offsets = np.load(OFFSETS)
    sources = set()
    for f, v in sorted(fam_var.items()):
        blob = export_lk(*EXT_BLOBS[f][:1], f, '', EXT_BLOBS[f][1], offsets) if v == 'ext' else \
            export_family(f, v, offsets)
        torch.save(blob, out / f'fam_{f}.pt')
        sources |= {s for s, _ in blob['sources']}
    for p in comps.values():
        for q in p.get('preds', []):
            sources.add({'pred0': 'cnn0', 'pred1': 'cnn1'}.get(q) or p['ridge'][q]['src'])
    torch.save({c: p for c, p in comps.items()}, out / 'components.pt')
    # feature-source artifacts
    if 'stats' in sources:
        torch.save({k: T(v, torch.float32) for k, v in np.load(EDGES['edges_stats']).items()}, out / 'edges_stats.pt')
    for c in ('cnn0', 'cnn1'):
        if c in sources:
            st = torch.load(CNN_CKPT[c], map_location='cpu', weights_only=False)
            torch.save(dict(net=compact(st['net']), widths=[int(w) for w in st['widths']]), out / f'{c}.pt')
    if sources & {'den_r', 'den_s', 'den_b'}:
        st = torch.load(UNET, map_location='cpu', weights_only=False)
        pg = torch.Generator(); pg.manual_seed(12345)
        probe = (torch.randint(0, 2, (1424, 176), generator=pg) * 2 - 1).to(torch.int8)   # extract_den2 probe
        torch.save(dict(net=compact(st['net']), widths=[int(w) for w in st['widths']], probe=probe), out / 'unet.pt')
        torch.save({k: T(v, torch.float32) for k, v in np.load(EDGES['edges_denoised']).items()},
                   out / 'edges_denoised.pt')
    if 'oe_head' in sources:
        st = torch.load(OE_CKPT, map_location='cpu', weights_only=False)
        torch.save(dict(net=compact(st['net']), nfam=int(st['net']['fam.1.weight'].shape[0] - 1)), out / 'oe.pt')
    if 'difa' in sources:
        st = torch.load(DUNET, map_location='cpu', weights_only=False)     # fp32 (GroupNorm net, kept exact)
        torch.save(dict(cfg=dict(st['cfg']), ema={k: v.cpu() for k, v in st['ema'].items()}), out / 'dunet.pt')
    if sources & {'stx_nois', 'stx_tail', 'stx_bisp'}:
        torch.save({k: T(v, torch.float32) for k, v in np.load(EDGES['edges_nois']).items()}, out / 'edges_nois.pt')
    if sources & {'stx_scat', 'stx_prof', 'stx_pkcl', 'stx_ipow'}:
        torch.save({k: T(v, torch.float32) for k, v in np.load(EDGES['edges_fam']).items()}, out / 'edges_fam.pt')
    cfg = dict(weights=weights, variants=var, batch=args.batch, profile_chunk=args.profile_chunk,
               families=sorted(fam_var), sources=sorted(sources))
    (out / 'config.json').write_text(json.dumps(cfg, indent=1) + '\n')
    print('components', {c: p['kind'] for c, p in comps.items()})
    print('families', sorted(fam_var), 'sources', sorted(sources))

    print('CPU parity (stored per-map family outputs -> packaged component math):')
    z = parity_cpu(comps, weights, fam_var, args.name, args.ref_components)
    np.save(out.parent / '_cache' / f'cpu_recompose_{args.name}.npy', z)
    if args.ref:
        r = np.load(args.ref)
        print(f'  recipe z vs {args.ref}: max abs diff {np.abs(z - r).max():.2e}')
    for p in out.rglob('__pycache__'):
        shutil.rmtree(p)
    shutil.make_archive(str(out), 'zip', root_dir=out)
    size = sum(p.stat().st_size for p in out.iterdir()) / 1e6
    print(f'packaged {out} ({size:.1f} MB) and {out}.zip')


if __name__ == '__main__':
    main()
