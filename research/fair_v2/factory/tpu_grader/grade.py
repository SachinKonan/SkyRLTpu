"""TPU (JAX) port of factory/grade.py: marginal leave-cosmology-out (LCO) gain of one JAX feature family.

Self-contained (numpy + jax + stdlib only; no torch, scipy or sklearn). One grader per v5p chip.
Importable worker API (one persistent process per chip):

    from tpu_grader import grade as G
    G.init(chip=0, stack='v2')                  # sets the per-chip TPU env, imports jax, loads data + caches
    res = G.grade_source(src, mode='rl')        # same dict as result.json (incl. the hidden block)

CLI (one process, one or more candidates):

    python -m tpu_grader.grade <candidate.py> [...] --out <dir> [--mode rl|screen|full] [--chip k] [--stack v1|v2]

The candidate file defines

    def features(maps, mask):   # maps: jnp.float32 (B, 1424, 176) on the TPU, zero outside the footprint
        ...                     # mask: jnp.bool_ (1424, 176)
        return feats            # (B, d) jnp array, finite, a pure per-map function; jit-compatible

Pipeline and reward are those of grade.py (MOMO_HILLCLIMB_ENV.md section 3); the likelihood member is a JAX
port of jlik.JLikelihood (model.Likelihood with the jitter mask). Every input comes from one root
(init(root=...) or $FAIR_FACTORY_ROOT, default /dev/shm/fair/factory), staged by factory/tpu_prep.py +
tpu_stage.sh: noisy/*.npy frozen noisy dumps (shift dumps hold only the held-out audit rows), meta.npz
(labels, per-fold masks, held rows, footprint mask, shift order) and cache_<stack>/fold{k}.npz + stack.json
(stack members and consistency views per fold). The public test file is never read.
"""
import argparse
import gc
import hashlib
import json
import math
import os
import signal
import statistics
import sys
import threading
import time
import traceback
import types
from pathlib import Path

import numpy as np

ROOT = Path(os.environ.get('FAIR_FACTORY_ROOT', '/dev/shm/fair/factory'))
FOLDS = (0, 1, 2)
H, W = 1424, 176
HIDDEN = ['baryDMO', 'mult-0.05', 'noise_t5', 'aniso0.7', 'ia0.2', 'peakcut0.3']
# stack v1 = v3-12 without the outlier-exposure head; v2 adds dirpower (see cache_v2/stack.json)
STACK_V1 = {'Sd': .5, 'C': 1, 'Dr': 2.5, 'xSL': 1, 'consP8': .5, 'consRplus': 3, 'N2': .5, 'PsBh': .5, 'SS': 1}
VIEWS_V1 = ['b0', 'b1', 'b2', 'b3', 'cnn', 'cnn1', 'pred0', 'pred1', 'dog', 'den_r', 'den_sd', 'den_b']
MIS_W = {'screen': [0], 'rl': [0, .5, 1, 2], 'full': [0, .5, 1, 2]}
VOTES = {'screen': ['none', 'ridge'], 'rl': ['none', 'ridge', 'prof'], 'full': ['none', 'ridge', 'prof']}
# rows: fit (step, views) and shift (step over the held-out audit rows); calib (2 views) and audit are always whole
# rl = the full computation with a tighter runtime budget: on a TPU the likelihood refits are cheap, and a
# one-fit-view variant (fit=(1, 1), ~3.5 s faster) ranked dirpower8 barely above random features (README_TPU.md)
ROWS = {'full': dict(fit=(1, 2), shift=1), 'screen': dict(fit=(5, 1), shift=3), 'rl': dict(fit=(1, 2), shift=1)}
LIK = {'full': dict(steps=3000, reg_steps=3000, prof_steps=150, restarts=3),
       'rl': dict(steps=3000, reg_steps=3000, prof_steps=150, restarts=3)}
BUDGET = {'rl': 15.0, 'screen': 60.0, 'full': 60.0}    # max TPU seconds per 10k maps (dirpower: 1.2, dog: ~3.5)
LIK_COMMON = dict(width=256, depth=3, batch=2048, lr=2e-3, wd=1e-5, jitter=0.1, jm=(0., 0., 1., 1., .5), eig_floor=0.02)
TAIL_PENALTY = 0.01
DZ_SIGMA = 0.022
BATCH = 64
CHUNK = 1024
ALPHAS = [1e-1, 1, 10, 100, 1e3, 1e4]
CLEAR_EVERY = 64      # jax.clear_caches() every N grades (memory was flat over 24 grades without it)

S = types.SimpleNamespace(ready=False)   # worker state (init)


def log(msg):
    if S.__dict__.get('verbose', True):
        print(f'[{time.strftime("%H:%M:%S")}] {msg}', flush=True)


# ============================================================================ numpy helpers (from the harness)
def design(theta, t_mean, t_std):
    z = (theta - t_mean) / t_std
    cols = [np.ones(len(z))] + [z[:, i] for i in range(5)] + [z[:, i] * z[:, j] for i in range(5) for j in range(i, 5)]
    return np.stack(cols, 1)


class Standardizer:
    """model.Standardizer: normal location/scale of a raw score as a quadratic function of theta_hat."""
    def fit(self, raw, theta, t_mean, t_std, ridge=1.0):
        self.t_mean, self.t_std = t_mean, t_std
        X = design(theta, t_mean, t_std)
        A = X.T @ X + ridge * np.eye(X.shape[1])
        self.loc = np.linalg.solve(A, X.T @ raw)
        res = raw - X @ self.loc
        self.scale = np.linalg.solve(A, X.T @ np.log(res ** 2 + 1e-12))
        self.norm = 1.0
        self.norm = float(self(raw, theta).std())
        return self

    def __call__(self, raw, theta):
        X = design(theta, self.t_mean, self.t_std)
        return (raw - X @ self.loc) / np.exp(0.5 * (X @ self.scale)) / self.norm


def phase2_score(normal, anomalous):
    """model.phase2_score with sklearn.roc_curve (drop_intermediate=True) reimplemented in numpy."""
    y = np.r_[np.zeros(len(normal)), np.ones(len(anomalous))]
    s = np.r_[normal, anomalous]
    order = np.argsort(s, kind='mergesort')[::-1]
    s, y = s[order], y[order]
    idx = np.r_[np.where(np.diff(s))[0], y.size - 1]
    tps = np.cumsum(y)[idx]
    fps = 1 + idx - tps
    if len(fps) > 2:
        keep = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps, tps = fps[keep], tps[keep]
    tps, fps = np.r_[0, tps], np.r_[0, fps]
    fpr, tpr = fps / fps[-1], tps / tps[-1]
    grid = np.logspace(-3, math.log10(0.05), 100)
    return float(np.mean(np.interp(grid, fpr, tpr)))


def standardize(raw, theta):
    """analyze.standardize: Standardizer fit on calib; theta normalized by the global fit3 label moments."""
    st = Standardizer().fit(raw['calib'], theta['calib'] * S.TS + S.TM, S.TM, S.TS)
    return {k: st(raw[k], theta[k] * S.TS + S.TM) for k in raw}


def mean_of(*zs, w=None):
    w = np.ones(len(zs)) if w is None else np.asarray(w)
    return {k: sum(wi * zi[k] for wi, zi in zip(w, zs)) / w.sum() for k in zs[0]}


def mahal(vecs, ridge=1e-3):
    c = vecs['calib']
    mu = c.mean(0)
    C = np.cov(c - mu, rowvar=False)
    C = C + ridge * np.trace(C) / len(C) * np.eye(len(C))
    Ci = np.linalg.inv(C)
    return {k: np.einsum('ij,jk,ik->i', v - mu, Ci, v - mu) for k, v in vecs.items()}


def cons(TH, fs, names):
    vec, cen = {}, {}
    for n in names:
        A = np.stack([TH[f][n] for f in fs], 1)
        vec[n] = (A - A.mean(1, keepdims=True))[:, 1:].reshape(len(A), -1)
        cen[n] = A.mean(1)
    m2 = mahal(vec)
    return standardize({n: np.log(m2[n] + 1.0) for n in names}, cen)


def ridge_fit(Xf, yf, Xc, yc):
    mu, sd = Xf.mean(0), Xf.std(0) + 1e-8
    A = (Xf - mu) / sd
    ym = yf.mean(0)
    G = A.T @ A
    b = A.T @ (yf - ym)
    best = None
    for a in ALPHAS:
        Wt = np.linalg.solve(G + a * len(A) / 1000 * np.eye(len(G)), b)
        mse = np.mean((((Xc - mu) / sd) @ Wt + ym - yc) ** 2)
        if best is None or mse < best[0]:
            best = (mse, a, Wt)
    _, a, Wt = best
    return lambda X: ((X - mu) / sd) @ Wt + ym, a, best[0]


def sub(dct, take):
    return {n: (v[take] if n.startswith('sh_') else v) for n, v in dct.items()}


def metrics(z):
    sh = {s: phase2_score(z['out'], z['sh_' + s]) for s in S.SHIFTS}
    t = np.quantile(z['in'], [.99, .995, .999])
    exc = [float(np.mean(z['out'] > x)) * 100 for x in t]
    return sh, exc


# ================================================================================================ init
def init(chip=None, stack='v2', root=None, verbose=True):
    """Pin this process to one TPU chip (chip=None: leave the environment alone), import jax, load data."""
    global ROOT
    S.verbose = verbose
    if root is not None:
        ROOT = Path(root)
    if chip is not None and 'jax' not in sys.modules:
        port = 8476 + int(chip)
        os.environ.update(TPU_VISIBLE_CHIPS=str(chip), TPU_PROCESS_BOUNDS='1,1,1', TPU_CHIPS_PER_PROCESS_BOUNDS='1,1,1',
                          TPU_PROCESS_PORT=str(port), TPU_PROCESS_ADDRESSES=f'localhost:{port}', CLOUD_TPU_TASK_ID='0',
                          JAX_PLATFORMS=os.environ.get('JAX_PLATFORMS', 'tpu'))
        os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE', 'false')
        for k in ('JAX_COORDINATOR_ADDRESS', 'TPU_MULTIHOST_BACKEND'):
            os.environ.pop(k, None)
    t0 = time.time()
    import jax
    import jax.numpy as jnp
    S.jax, S.jnp = jax, jnp
    jax.config.update('jax_default_matmul_precision', 'highest')
    cache = os.environ.get('FAIR_JAX_CACHE', str(ROOT / 'jax_cache'))
    if cache not in ('', '0', 'off'):
        # persistent compilation cache for the likelihood programs (same for every candidate in a dim bucket),
        # so a fresh process per grade skips their ~20 s compile; candidate programs are never written
        jax.config.update('jax_compilation_cache_dir', cache)
        jax.config.update('jax_persistent_cache_min_entry_size_bytes', 0)
        S.jax_cache = cache
    _cache_writes(False)
    S.device = jax.devices()[0]
    M = np.load(ROOT / 'meta.npz')
    S.meta = {k: M[k] for k in M.files}
    S.SHIFTS = [str(s) for s in S.meta['shifts']]
    S.VISIBLE = [s for s in S.SHIFTS if s not in HIDDEN]
    S.NAMES = ['calib', 'in', 'out'] + ['sh_' + s for s in S.SHIFTS]
    S.TM, S.TS = S.meta['labels_fit'].mean(0), S.meta['labels_fit'].std(0)
    mask = S.meta['mask'].astype(bool)
    flat = np.nonzero(mask.ravel())[0]
    inv = np.full(H * W, len(flat), np.int32)
    inv[flat] = np.arange(len(flat), dtype=np.int32)
    S.nvalid = len(flat)
    S.inv = jnp.asarray(inv)
    S.mask = jnp.asarray(mask)
    S.held = S.meta['held_rows']
    S.mm = {}
    for p in sorted((ROOT / 'noisy').glob('*.npy')):
        S.mm[p.stem] = np.load(p, mmap_mode='r')
    S.stacks = {}
    load_stack(stack)
    S.default_stack = stack
    S.lik_fns = {}
    S.ready = True
    log(f'init: device {S.device} stack {stack} {len(S.mm)} dumps, {time.time()-t0:.1f}s')
    return S.device


def _cache_writes(on):
    """Allow persistent-cache writes only while compiling the grader's own (likelihood) programs."""
    S.jax.config.update('jax_persistent_cache_min_compile_time_secs', 0.0 if on else 1e9)


def load_stack(name):
    if name in S.stacks:
        return S.stacks[name]
    d = ROOT / f'cache_{name}'
    caches = {}
    for k in FOLDS:
        C = dict(comp={}, TH={})
        with np.load(d / f'fold{k}.npz') as z:
            for key in z.files:
                part, member, setname = key.split('/', 2)
                C[part].setdefault(member, {})[setname] = z[key]
        caches[k] = C
    spec = json.loads((d / 'stack.json').read_text())
    st = dict(caches=caches, stack=spec['stack'], views=spec['views'], name=name)
    S.stacks[name] = st
    return st


# =========================================================================================== candidate
def compile_candidate(src, tag):
    jax, jnp = S.jax, S.jnp
    mod = types.ModuleType(f'candidate_{tag}')
    mod.__file__ = f'<candidate {tag}>'
    exec(compile(src, mod.__file__, 'exec'), mod.__dict__)
    fn = getattr(mod, 'features', None)
    if not callable(fn):
        raise ValueError('candidate defines no features(maps, mask)')
    mask, inv = S.mask, S.inv

    def run(rows16):
        ext = jnp.concatenate([rows16.astype(jnp.float32), jnp.zeros((rows16.shape[0], 1), jnp.float32)], 1)
        maps = jnp.take(ext, inv, axis=1).reshape(rows16.shape[0], H, W)
        f = fn(maps, mask)
        f = jnp.asarray(f)
        if f.ndim != 2 or f.shape[0] != rows16.shape[0]:
            raise ValueError(f'features must return (B, d); got {f.shape} for B={rows16.shape[0]}')
        return f.astype(jnp.float32)
    return mod, jax.jit(run)


class Runner:
    def __init__(self, f):
        self.f, self.dev_s, self.nmaps = f, 0.0, 0

    def call(self, rows16):
        """rows16: (n <= BATCH, nvalid) float16 numpy -> (n, d) float32 numpy (zero-padded batch)."""
        n = len(rows16)
        if n < BATCH:
            rows16 = np.concatenate([rows16, np.zeros((BATCH - n, S.nvalid), np.float16)])
        return np.asarray(self.f(S.jax.device_put(np.ascontiguousarray(rows16), S.device)))[:n]

    def run_rows(self, arr, rows):
        """Features of arr[rows]: chunked host reads with one-chunk prefetch, batches dispatched asynchronously."""
        jax = S.jax
        chunks = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
        out = []
        box = {}

        def read(r):
            if len(r) and r[-1] - r[0] == len(r) - 1:
                return np.ascontiguousarray(arr[r[0]:r[-1] + 1])
            return np.asarray(arr[r])

        def prefetch(j):
            box[j] = read(chunks[j])
        th = threading.Thread(target=prefetch, args=(0,)); th.start()
        for j in range(len(chunks)):
            th.join()
            block = box.pop(j)
            if j + 1 < len(chunks):
                th = threading.Thread(target=prefetch, args=(j + 1,)); th.start()
            t = time.time()
            pend = []
            for b in range(0, len(block), BATCH):
                x = block[b:b + BATCH]
                n = len(x)
                if n < BATCH:
                    x = np.concatenate([x, np.zeros((BATCH - n, S.nvalid), np.float16)])
                pend.append((self.f(jax.device_put(x, S.device)), n))
            for f, n in pend:
                out.append(np.asarray(f)[:n])
            self.dev_s += time.time() - t
            self.nmaps += len(block)
        return np.concatenate(out)


def contract(R, budget):
    """Hard checks with one compiled batch shape (BATCH): finite, d in [1, 4096], per-map (other maps in the
    batch replaced, zero padding, shuffled order), determinism; runtime on 512 audit maps."""
    arr = S.mm['audit_v0']
    X = np.asarray(arr[:BATCH])
    info = {}
    try:
        t = time.time()
        a = R.call(X)
        info['compile_s'] = round(time.time() - t, 2)
        info['dim'] = int(a.shape[1])
        if not 1 <= a.shape[1] <= 4096:
            return False, dict(info, fail=f'feature dimension {a.shape[1]} outside [1, 4096]')
        if not np.isfinite(a).all():
            return False, dict(info, fail='non-finite features on normal maps')
        scale = np.abs(a).max(0) + 1e-6
        perm = np.random.default_rng(0).permutation(BATCH)
        shuf = R.call(X[perm])[np.argsort(perm)]
        Y = X.copy()
        Y[8:] = np.asarray(arr[BATCH:2 * BATCH - 8])
        other = R.call(Y)[:8]
        pad = R.call(X[:8])
        again = R.call(X)
        checks = (('per_map', other, a[:8], scale), ('padding', pad, a[:8], scale), ('shuffle', shuf, a, scale),
                  ('determinism', again, a, scale))
        for key, x, ref, sc in checks:
            rel = np.abs(x - ref) / sc
            info[f'{key}_max_rel'] = float(rel.max())
            info[f'{key}_frac_off'] = float((rel > 1e-3).mean())
        if max(info['per_map_frac_off'], info['padding_frac_off'], info['shuffle_frac_off']) > 0.01:
            return False, dict(info, fail='not a per-map function (other maps in the batch, zero padding or '
                                          'shuffled order change a map\'s features)')
        if info['determinism_frac_off'] > 0.01:
            return False, dict(info, fail='not deterministic')
        R.run_rows(arr, np.arange(64, 128))                 # warm-up of the pipelined path
        R.dev_s, R.nmaps = 0.0, 0
        t = time.time()
        R.run_rows(arr, np.arange(128, 640))
        sec10k = (time.time() - t) / 512 * 10000
        R.dev_s, R.nmaps = 0.0, 0
        info['sec_per_10k'] = round(sec10k, 2)
        info['budget_sec_per_10k'] = budget
        if sec10k > budget:
            return False, dict(info, fail=f'runtime {sec10k:.0f}s per 10k maps exceeds budget {budget}s')
    except Exception as e:  # noqa: BLE001 - any candidate error is a contract failure
        return False, dict(info, fail=f'{type(e).__name__}: {str(e)[:2000]}')
    return True, info


def rowsets(mode):
    nf, nc, na = (len(S.meta[f'labels_{n}']) for n in ('fit', 'calib', 'audit'))
    fs, fv = ROWS[mode]['fit']
    return dict(fit=(np.arange(0, nf, fs), fv), calib=(np.arange(nc), 2), audit=(np.arange(na), 1),
                shift=(np.arange(0, len(S.held), ROWS[mode]['shift']), 1))


def pad_sizes(mode):
    """Padded fit-row count and profile-row count for this mode: the max over folds, rounded up."""
    rows = rowsets(mode)
    M = S.meta
    fitrows = rows['fit'][0]
    fv = rows['fit'][1]
    shrows = S.held[rows['shift'][0]]
    nfit, nprof = [], []
    for k in FOLDS:
        mf, mc, ma = M[f'mask{k}_fit'], M[f'mask{k}_calib'], M[f'mask{k}_audit']
        nfit.append(int((~mf[fitrows]).sum()))
        nsh = int(np.isin(np.nonzero(ma)[0], shrows).sum())
        nprof.append(2 * int((~mc).sum()) + int((~ma).sum()) + 2 * int(mc.sum()) + int(ma.sum())
                     + len(S.SHIFTS) * nsh)
    rnd = lambda v, m: int(math.ceil(v / m) * m)
    return rnd(max(nfit), 512), rnd(max(nprof), 512)


def extract(R, rows):
    F = {}
    jobs = [('fit',) + rows['fit'], ('calib',) + rows['calib'], ('audit',) + rows['audit']]
    jobs += [('shift_' + s,) + rows['shift'] for s in S.SHIFTS]
    for name, idx, views in jobs:
        parts = [R.run_rows(S.mm[f'{name}_v{v}'], idx) for v in range(views)]
        f = np.stack(parts, 1)
        if not np.isfinite(f).all():
            raise FloatingPointError(f'non-finite features in {name}')
        F[name] = f
    return F


# ================================================================================= JAX likelihood member
def onecycle(total, max_lr, pct=0.05, div=25., final_div=1e4, base_m=0.85, max_m=0.95):
    """torch OneCycleLR (cos annealing, cycle_momentum): lr and Adam beta1 used at optimizer step s."""
    init_lr = max_lr / div
    min_lr = init_lr / final_div
    e1, e2 = float(pct * total) - 1, total - 1
    cosa = lambda a, b, p: b + (a - b) / 2.0 * (math.cos(math.pi * p) + 1)
    lr, b1 = np.zeros(total), np.zeros(total)
    for s in range(total):
        if s <= e1:
            p = s / e1
            lr[s], b1[s] = cosa(init_lr, max_lr, p), cosa(max_m, base_m, p)
        else:
            p = (s - e1) / (e2 - e1)
            lr[s], b1[s] = cosa(max_lr, min_lr, p), cosa(base_m, max_m, p)
    return lr.astype(np.float32), b1.astype(np.float32)


DK_BUCKETS = (16, 32, 64, 128, 256, 512, 1024, 2048, 4096)


def dk_bucket(dk):
    return next(b for b in DK_BUCKETS if b >= dk)


def lik_fns(npad, V, dkp, steps, reg_steps, width, depth, batch, wd, jitter, jm):
    """Jitted training / scoring functions for one padded shape signature, compiled once per worker.

    Feature dimensions are padded to a bucket (dkp) and masked with dm (1 for the dk real dims), so every
    candidate with dk in the same bucket reuses the compiled programs; rows are padded to npad and masked.
    """
    key = ('lik', npad, V, dkp, steps, reg_steps, width, depth, batch)
    if key in S.lik_fns:
        return S.lik_fns[key]
    jax, jnp = S.jax, S.jnp
    jmv = jnp.asarray(jm, jnp.float32)

    def linit(k, fi, fo, bound):
        k1, k2 = jax.random.split(k)
        return (jax.random.uniform(k1, (fi, fo), jnp.float32, -1.0, 1.0) * bound,
                jax.random.uniform(k2, (fo,), jnp.float32, -1.0, 1.0) * bound)

    def net_init(k):
        ks = jax.random.split(k, depth + 2)
        sizes = [5] + [width] * depth
        return dict(body=[linit(ks[i], sizes[i], sizes[i + 1], 1.0 / math.sqrt(sizes[i])) for i in range(depth)],
                    mu=linit(ks[depth], width, dkp, 1.0 / math.sqrt(width)),
                    lv=linit(ks[depth + 1], width, dkp, 1.0 / math.sqrt(width)))

    def net(p, t):
        h = t
        for Wt, b in p['body']:
            h = jax.nn.silu(h @ Wt + b)
        return h @ p['mu'][0] + p['mu'][1], jnp.clip(h @ p['lv'][0] + p['lv'][1], -12, 6)

    def reg_init(k, dkf):
        ks = jax.random.split(k, 3)
        return [linit(ks[0], dkp, 512, 1.0 / jnp.sqrt(dkf)), linit(ks[1], 512, 512, 1.0 / math.sqrt(512)),
                linit(ks[2], 512, 5, 1.0 / math.sqrt(512))]

    def reg(p, x):
        h = jax.nn.silu(x @ p[0][0] + p[0][1])
        h = jax.nn.silu(h @ p[1][0] + p[1][1])
        return h @ p[2][0] + p[2][1]

    def adamw(p, g, m, v, lr, b1, step, wd_, b2=0.999, eps=1e-8):
        bc1, bc2 = 1 - b1 ** step, 1 - b2 ** step
        m = jax.tree.map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)
        v = jax.tree.map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ * g_, v, g)
        p = jax.tree.map(lambda p_, m_, v_: p_ * (1 - lr * wd_) - (lr / bc1) * m_ / (jnp.sqrt(v_) / jnp.sqrt(bc2) + eps),
                         p, m, v)
        return p, m, v

    @jax.jit
    def train(x, t, n, dm, key, lrs, b1s):
        key, k0 = jax.random.split(key)
        p = net_init(k0)
        z = jax.tree.map(jnp.zeros_like, p)
        denom = batch * jnp.sum(dm)

        def loss_fn(p, xb, tb):
            mu, lv = net(p, tb)
            r = xb - mu
            return 0.5 * jnp.sum((r * r * jnp.exp(-lv) + lv) * dm) / denom

        def step(c, inp):
            p, m, v, key = c
            lr, b1, s = inp
            key, k1, k2, k3 = jax.random.split(key, 4)
            i = jax.random.randint(k1, (batch,), 0, n)
            vi = jax.random.randint(k2, (batch,), 0, V)
            ti = t[i] + jitter * jax.random.normal(k3, (batch, 5)) * jmv
            loss, g = jax.value_and_grad(loss_fn)(p, x[i, vi], ti)
            p, m, v = adamw(p, g, m, v, lr, b1, s, wd)
            return (p, m, v, key), loss
        (p, _, _, key), losses = jax.lax.scan(step, (p, z, z, key),
                                              (lrs, b1s, jnp.arange(1, steps + 1, dtype=jnp.float32)))
        return p, key, losses[-1]

    @jax.jit
    def residual_stats(p, x, t, n, dm, pos_lo, pos_hi, frac, gk):
        """Residuals of the fit rows -> Gram matrix, per-dim quantile knots, copula Gram matrix."""
        mu, lv = net(p, t)
        u = ((x - mu[:, None]) * jnp.exp(-0.5 * lv)[:, None] * dm).reshape(-1, dkp)
        valid = (jnp.arange(u.shape[0]) < n * V)[:, None]
        uz = jnp.where(valid, u, 0.0)
        G = uz.T @ uz
        srt = jnp.sort(jnp.where(valid, u, jnp.inf), axis=0)
        q = (srt[pos_lo] * (1 - frac[:, None]) + srt[pos_hi] * frac[:, None]).T          # (dkp, K)
        q = jnp.where(dm[:, None] > 0, q, jnp.arange(q.shape[1], dtype=jnp.float32)[None])   # padded dims: dummy knots
        g = jnp.where(valid, copula(u, q, gk) * dm, 0.0)
        return G, q, g.T @ g

    def copula(u, q, gk):
        K = q.shape[1]
        ut = u.T
        i = jnp.clip(jax.vmap(lambda a, v: jnp.searchsorted(a, v, side='left'))(q, ut), 1, K - 1)
        lo = jnp.take_along_axis(q, i - 1, 1)
        hi = jnp.take_along_axis(q, i, 1)
        w = jnp.clip((ut - lo) / jnp.maximum(hi - lo, 1e-9), 0, 1)
        g = gk[i - 1] + w * (gk[i] - gk[i - 1])
        slope_hi = (gk[-1] - gk[-11]) / jnp.maximum(q[:, -1] - q[:, -11], 1e-6)
        slope_lo = (gk[10] - gk[0]) / jnp.maximum(q[:, 10] - q[:, 0], 1e-6)
        g = jnp.where(ut > q[:, -1:], gk[-1] + slope_hi[:, None] * (ut - q[:, -1:]), g)
        g = jnp.where(ut < q[:, :1], gk[0] + slope_lo[:, None] * (ut - q[:, :1]), g)
        return jnp.clip(g, -8, 8).T

    @jax.jit
    def train_reg(x, t, n, dkf, key):
        key, k0 = jax.random.split(key)
        p = reg_init(k0, dkf)
        z = jax.tree.map(jnp.zeros_like, p)

        def step(c, s):
            p, m, v, key = c
            key, k1, k2 = jax.random.split(key, 3)
            i = jax.random.randint(k1, (batch,), 0, n)
            vi = jax.random.randint(k2, (batch,), 0, V)
            loss, g = jax.value_and_grad(lambda p: jnp.mean((reg(p, x[i, vi]) - t[i]) ** 2))(p)
            p, m, v = adamw(p, g, m, v, 1e-3, 0.9, s, 1e-5)
            return (p, m, v, key), loss
        (p, _, _, key), losses = jax.lax.scan(step, (p, z, z, key), jnp.arange(1, reg_steps + 1, dtype=jnp.float32))
        return p, losses[-1]

    fns = dict(train=train, residual_stats=residual_stats, train_reg=train_reg, net=net, reg=reg, copula=copula)
    S.lik_fns[key] = fns
    return fns


def profile_fn(fns, nprof, dkp, prof_steps, restarts):
    key = ('profile', nprof, dkp, prof_steps, restarts, id(fns))
    if key in S.lik_fns:
        return S.lik_fns[key]
    jax, jnp = S.jax, S.jnp
    net, reg, copula = fns['net'], fns['reg'], fns['copula']

    @jax.jit
    def prof(p, rp, x, dm, Rinv, Rc_inv, q, gk, lo, hi, cosmo, ncos, hull_r, tm4, ts4, offsets):
        cvalid = jnp.arange(cosmo.shape[0]) < ncos

        def energy(t):
            mu, lv = net(p, t)
            u = (x - mu) * jnp.exp(-0.5 * lv) * dm
            return jnp.sum((u @ Rinv) * u, 1) + jnp.sum(lv * dm, 1)

        def prior(t):
            dz = t[:, 4] * ts4 + tm4
            d2 = jnp.sum((t[:, None, :2] - cosmo[None]) ** 2, -1)
            d2 = jnp.where(cvalid[None], d2, jnp.inf)
            dmin = jnp.sqrt(jnp.maximum(jnp.min(d2, 1), 0.0))
            return (dz / DZ_SIGMA) ** 2 + 50.0 * jax.nn.relu(dmin - hull_r) ** 2

        obj = lambda t: energy(t) + prior(t)
        t0 = jnp.clip(reg(rp, x), lo, hi)
        best_l = jnp.full((x.shape[0],), jnp.inf)
        best_t = t0
        for r in range(restarts):
            t = jnp.clip(t0 + offsets[r], lo, hi)

            def step(c, s):
                t, m, v = c
                g = jax.grad(lambda t: jnp.sum(obj(t)))(t)
                m = 0.9 * m + 0.1 * g
                v = 0.999 * v + 0.001 * g * g
                t = t - (0.05 / (1 - 0.9 ** s)) * m / (jnp.sqrt(v) / jnp.sqrt(1 - 0.999 ** s) + 1e-8)
                return (jnp.clip(t, lo, hi), m, v), None
            (t, _, _), _ = jax.lax.scan(step, (t, jnp.zeros_like(t), jnp.zeros_like(t)),
                                        jnp.arange(1, prof_steps + 1, dtype=jnp.float32))
            l = obj(t)
            better = l < best_l
            best_l = jnp.where(better, l, best_l)
            best_t = jnp.where(better[:, None], t, best_t)
        mu, lv = net(p, best_t)
        g = copula((x - mu) * jnp.exp(-0.5 * lv) * dm, q, gk) * dm
        return best_l, best_t, jnp.sum((g @ Rc_inv) * g, 1)
    S.lik_fns[key] = prof
    return prof


def oas_correlation(G, n):
    """model.oas_correlation from the Gram matrix (float64)."""
    p = len(G)
    s = G / n
    d = np.maximum(np.sqrt(np.diag(s)), 1e-6)
    s = s / d[:, None] / d[None, :]
    tr2 = (s * s).sum()
    rho = ((1 - 2 / p) * tr2 + p ** 2) / ((n + 1 - 2 / p) * (tr2 - p ** 2 / p))
    rho = float(np.clip(rho, 0, 1))
    return (1 - rho) * s + rho * np.eye(p), rho


def lik_member(Sd, Lfit, mode, npad, nprof, seed=0):
    """JAX JLikelihood: fit on Sd['fit'] (N, V, d), profile every set in S.NAMES. Returns (cop, th, info).

    npad / nprof: fixed padded row counts for this mode (same for every fold and candidate), so the
    compiled programs are reused across folds and grades."""
    jax, jnp = S.jax, S.jnp
    cfg = dict(LIK_COMMON, **LIK[mode])
    tt = {}
    t0 = time.time()
    feats, theta = Sd['fit'], Lfit
    N, V, d = feats.shape
    flat = feats.reshape(N * V, d).astype(np.float64)
    f_mean, f_std = flat.mean(0), flat.std(0) + 1e-8
    keep = f_std > 1e-6
    dk = int(keep.sum())
    if dk == 0:
        raise ValueError('every feature is constant on the fit maps')
    dkp = dk_bucket(dk)
    t_mean, t_std = theta.mean(0), theta.std(0)
    t_lo, t_hi = (theta.min(0) - t_mean) / t_std, (theta.max(0) - t_mean) / t_std
    cos = (np.unique(theta[:, :2], axis=0) - t_mean[:2]) / t_std[:2]
    dd = np.sqrt(((cos[:, None] - cos[None]) ** 2).sum(-1)) + 1e9 * np.eye(len(cos))
    hull_r = float(np.median(dd.min(1)))
    cosp = np.zeros((128, 2), np.float32)
    cosp[:len(cos)] = cos
    assert N <= npad and len(cos) <= 128
    x = np.zeros((npad, V, dkp), np.float32)
    x[:N, :, :dk] = ((feats - f_mean) / f_std)[..., keep]
    t = np.zeros((npad, 5), np.float32)
    t[:N] = (theta - t_mean) / t_std
    dm = np.zeros(dkp, np.float32)
    dm[:dk] = 1
    fns = lik_fns(npad, V, dkp, cfg['steps'], cfg['reg_steps'], cfg['width'], cfg['depth'], cfg['batch'], cfg['wd'],
                  cfg['jitter'], cfg['jm'])
    xj, tj, dmj = jnp.asarray(x), jnp.asarray(t), jnp.asarray(dm)
    lrs, b1s = onecycle(cfg['steps'], cfg['lr'])
    key = jax.random.PRNGKey(seed)
    tt['prep'] = time.time() - t0
    t0 = time.time()
    p, key, loss = fns['train'](xj, tj, N, dmj, key, jnp.asarray(lrs), jnp.asarray(b1s))
    loss = float(loss)
    tt['train'] = time.time() - t0
    t0 = time.time()
    n = N * V
    probs = np.linspace(0.0, 1.0, 1001)
    probs = (probs * (n - 1) + 0.5) / n
    pos = probs * (n - 1)
    lo_i = np.floor(pos).astype(np.int32)
    hi_i = np.minimum(lo_i + 1, n - 1).astype(np.int32)
    fr = (pos - lo_i).astype(np.float32)
    gk = np.array([statistics.NormalDist().inv_cdf(float(pp)) for pp in probs], np.float32)
    gkj = jnp.asarray(gk)
    G, q, Gc = fns['residual_stats'](p, xj, tj, N, dmj, jnp.asarray(lo_i), jnp.asarray(hi_i), jnp.asarray(fr), gkj)
    R, rho = oas_correlation(np.asarray(G, np.float64)[:dk, :dk], n)
    Rinv = np.eye(dkp, dtype=np.float32)
    Rinv[:dk, :dk] = np.linalg.inv(R)
    Rc, rho_c = oas_correlation(np.asarray(Gc, np.float64)[:dk, :dk], n)
    ev, evec = np.linalg.eigh(Rc)
    ev = np.maximum(ev, cfg['eig_floor'])
    Rc_inv = np.eye(dkp, dtype=np.float32)
    Rc_inv[:dk, :dk] = (evec / ev) @ evec.T
    tt['cov'] = time.time() - t0
    t0 = time.time()
    rp, rloss = fns['train_reg'](xj, tj, N, float(dk), key)
    rloss = float(rloss)
    tt['reg'] = time.time() - t0
    t0 = time.time()
    sizes = {nm: Sd[nm].shape[0] * Sd[nm].shape[1] for nm in S.NAMES}
    tot = sum(sizes.values())
    assert tot <= nprof
    X = np.zeros((nprof, dkp), np.float32)
    o = 0
    for nm in S.NAMES:
        X[o:o + sizes[nm], :dk] = ((Sd[nm].reshape(-1, d) - f_mean) / f_std)[:, keep]
        o += sizes[nm]
    offs = np.zeros((cfg['restarts'], 1, 5), np.float32)
    rk = np.random.default_rng(seed)
    for r in range(1, cfg['restarts']):
        offs[r] = 0.5 * rk.standard_normal((1, 5))
    prof = profile_fn(fns, nprof, dkp, cfg['prof_steps'], cfg['restarts'])
    _, bt, cop = prof(p, rp, jnp.asarray(X), dmj, jnp.asarray(Rinv), jnp.asarray(Rc_inv), q, gkj,
                      jnp.asarray(t_lo, jnp.float32), jnp.asarray(t_hi, jnp.float32), jnp.asarray(cosp), len(cos),
                      hull_r, float(t_mean[4]), float(t_std[4]), jnp.asarray(offs))
    bt, cop = np.asarray(bt, np.float64), np.asarray(cop, np.float64)
    tt['profile'] = time.time() - t0
    copd, th = {}, {}
    o = 0
    for nm in S.NAMES:
        th[nm] = bt[o:o + sizes[nm]].astype(np.float32)        # normalized by the fold's own label moments
        copd[nm] = cop[o:o + sizes[nm]]
        o += sizes[nm]
    info = dict(nll=round(loss, 4), rho=round(rho, 4), rho_c=round(rho_c, 4), reg_mse=round(rloss, 4),
                dk=dk, dk_bucket=dkp, n_fit=n, seconds={k: round(v, 2) for k, v in tt.items()})
    return copd, th, info


# ============================================================================================ folds
def fold_inputs(F, rows, k):
    M = S.meta
    mf, mc, ma = M[f'mask{k}_fit'], M[f'mask{k}_calib'], M[f'mask{k}_audit']
    fitrows = rows['fit'][0]
    mfr = mf[fitrows]
    Lfit_all = M['labels_fit']
    tm, ts = Lfit_all[~mf].mean(0), Lfit_all[~mf].std(0)
    d = F['fit'].shape[-1]
    Sd = dict(fit=F['fit'][~mfr], calib=F['calib'][~mc], out=np.concatenate(
        [F['calib'][mc].reshape(-1, 1, d), F['audit'][ma]], 0))
    Sd['in'] = F['audit'][~ma]
    shrows = S.held[rows['shift'][0]]                 # audit row ids of the computed shift rows (sorted)
    heldk = np.nonzero(ma)[0]
    take = np.isin(heldk, shrows)
    pos = np.searchsorted(shrows, heldk[take])
    for s in S.SHIFTS:
        Sd['sh_' + s] = F['shift_' + s][pos]
    Lcal = M['labels_calib'][~mc]
    return Sd, Lfit_all[fitrows][~mfr], Lcal, tm, ts, take


def ridge_vote(Sd, Lfit, Lcal, tm, ts):
    d = Sd['fit'].shape[-1]
    Xf = Sd['fit'].reshape(-1, d).astype(np.float64)
    yf = np.repeat((Lfit - tm) / ts, Sd['fit'].shape[1], 0)
    Xc = Sd['calib'].reshape(-1, d).astype(np.float64)
    yc = np.repeat((Lcal - tm) / ts, Sd['calib'].shape[1], 0)
    f, a, mse = ridge_fit(Xf, yf, Xc, yc)
    per = ((f(Xc) - yc) ** 2).mean(0)
    return {n: f(Sd[n].reshape(-1, d).astype(np.float64)).astype(np.float32) for n in S.NAMES}, \
        dict(alpha=a, calib_mse=round(float(mse), 4), calib_mse_per_param=np.round(per, 4).tolist())


# ============================================================================================= grading
class _Timeout(BaseException):           # not an Exception: the candidate-error handlers must not swallow it
    pass


def _mem_gb():
    """Private (anonymous) and file-backed resident memory; the /dev/shm dumps are shared file pages."""
    out = {}
    with open('/proc/self/status') as fh:
        for line in fh:
            if line.startswith(('RssAnon:', 'RssFile:', 'RssShmem:')):
                k, v = line.split(':')
                out[k] = int(v.split()[0]) / 1e6
    return out


def grade_source(src, mode='rl', stack=None, budget=None, timeout=None):
    """Grade one candidate source string. Never raises for candidate errors: returns ok=False with a reason."""
    assert S.ready, 'call init() first'
    stack = stack or S.default_stack
    budget = BUDGET[mode] if budget is None else budget
    t0 = time.time()
    sha = hashlib.sha256(src.encode()).hexdigest()[:16]
    res = dict(sha=sha, mode=mode, stack=stack, ok=False, reward=None)
    old = None
    if timeout and threading.current_thread() is threading.main_thread():
        def _alarm(*_):
            raise _Timeout(f'grade exceeded {timeout}s')
        old = signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, timeout)
    mod = R = None
    try:
        try:
            mod, f = compile_candidate(src, sha)
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001
            res['contract'] = dict(fail=f'import: {type(e).__name__}: {str(e)[:2000]}')
            return res
        R = Runner(f)
        ok, info = contract(R, budget)
        res['contract'] = info
        if not ok:
            return res
        try:
            _grade(R, mode, stack, res)
        except _Timeout:
            raise
        except Exception as e:  # noqa: BLE001
            res['ok'], res['reward'] = False, None
            res['contract']['fail'] = f'grading: {type(e).__name__}: {str(e)[:2000]}'
            res['traceback'] = traceback.format_exc()[-4000:]
        return res
    except _Timeout as e:
        res['contract'] = dict(res.get('contract') or {}, fail=f'timeout: {e}')
        return res
    finally:
        if old is not None:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
        rt = res.setdefault('runtime', {})
        rt['total_s'] = round(time.time() - t0, 2)
        del mod, R
        gc.collect()
        S.jax.config.update('jax_default_matmul_precision', 'highest')     # undo any candidate config change
        _cache_writes(False)
        m = _mem_gb()
        rt['host_private_gb'] = round(m.get('RssAnon', 0), 2)
        rt['host_shared_gb'] = round(m.get('RssFile', 0) + m.get('RssShmem', 0), 2)
        try:
            ms = S.device.memory_stats() or {}
            rt['device_gb_in_use'] = round(ms.get('bytes_in_use', 0) / 1e9, 2)
        except Exception:  # noqa: BLE001
            pass
        S.ngraded = S.__dict__.get('ngraded', 0) + 1
        if S.ngraded % CLEAR_EVERY == 0:     # safety valve; costs one likelihood recompile per dim bucket
            S.jax.clear_caches()


def _grade(R, mode, stack, res):
    S.jax.config.update('jax_default_matmul_precision', 'highest')         # the candidate may have changed it
    st = load_stack(stack)
    STACK, VIEWS0 = st['stack'], st['views']
    rows = rowsets(mode)
    t1 = time.time()
    F = extract(R, rows)
    t_extract = time.time() - t1
    t2 = time.time()
    parts, ridge_info, lik_info = {}, {}, {}
    npad, nprof = pad_sizes(mode)
    _cache_writes(True)                     # the candidate's programs are compiled by now
    for k in FOLDS:
        Sd, Lfit, Lcal, tm, ts, take = fold_inputs(F, rows, k)
        P = dict(take=take)
        P['ridge'], ridge_info[k] = ridge_vote(Sd, Lfit, Lcal, tm, ts)
        if mode in LIK:
            cop, th, lik_info[k] = lik_member(Sd, Lfit, mode, npad, nprof)
            P['prof'] = th
            P['mis'] = standardize(cop, th)
        parts[k] = P
    t_fit = time.time() - t2

    configs = [(w, v) for w in MIS_W[mode] for v in VOTES[mode] if (w, v) != (0, 'none')]
    configs = [c for c in configs if (c[0] == 0 or 'mis' in parts[0]) and (c[1] != 'prof' or 'prof' in parts[0])]
    base, cand, standalone = {}, {c: {} for c in configs}, {}
    for k in FOLDS:
        C, P = st['caches'][k], parts[k]
        take = P['take']
        comp = {m: sub(z, take) for m, z in C['comp'].items()}
        TH = {f: sub(v, take) for f, v in C['TH'].items()}
        comp['consRplus'] = cons(TH, VIEWS0, S.NAMES)
        base[k] = metrics(mean_of(*[comp[m] for m in STACK], w=list(STACK.values())))
        consv = {}
        for v in VOTES[mode]:
            if v != 'none' and v in P:
                TH['cand'] = P[v]
                consv[v] = cons(TH, VIEWS0 + ['cand'], S.NAMES)
        if 'mis' in P:
            standalone.setdefault('misfit', []).append(metrics(P['mis'])[0])
        for v in consv:
            standalone.setdefault(f'consRplus+{v}', []).append(metrics(consv[v])[0])
        standalone.setdefault('consRplus', []).append(metrics(comp['consRplus'])[0])
        for (w, v) in configs:
            c = dict(comp)
            Wt = dict(STACK)
            if v != 'none':
                c['consRplus'] = consv[v]
            if w > 0:
                c['X'] = P['mis']; Wt['X'] = w
            cand[(w, v)][k] = metrics(mean_of(*[c[m] for m in Wt], w=list(Wt.values())))

    def agg(sh, which):
        return float(np.mean([sh[s] for s in which]))

    def summarize(cfg):
        vis = [agg(cand[cfg][k][0], S.VISIBLE) - agg(base[k][0], S.VISIBLE) for k in FOLDS]
        hid = [agg(cand[cfg][k][0], HIDDEN) - agg(base[k][0], HIDDEN) for k in FOLDS]
        d_exc = np.mean([np.array(cand[cfg][k][1]) - np.array(base[k][1]) for k in FOLDS], 0)
        pen = TAIL_PENALTY * (max(0., d_exc[0]) + max(0., d_exc[1]))
        return dict(vis=vis, hid=hid, d_exc=d_exc.tolist(), pen=pen, reward=float(np.mean(vis) + min(vis) - pen))

    SS = {cfg: summarize(cfg) for cfg in configs}
    best = max(configs, key=lambda c: SS[c]['reward'])
    b = SS[best]
    per_shift = {s: round(float(np.mean([cand[best][k][0][s] - base[k][0][s] for k in FOLDS])), 4) for s in S.SHIFTS}
    exc_b = np.mean([base[k][1] for k in FOLDS], 0)
    exc_c = np.mean([cand[best][k][1] for k in FOLDS], 0)
    sa = {name: round(float(np.mean([agg(x, S.VISIBLE) for x in v])), 4) for name, v in standalone.items()}
    res.update(
        ok=True, reward=round(b['reward'], 5),
        config=dict(misfit_weight=best[0], vote=best[1]),
        visible=dict(
            shifts=S.VISIBLE,
            per_fold_gain=[round(x, 4) for x in b['vis']], mean_gain=round(float(np.mean(b['vis'])), 4),
            base_mean=round(float(np.mean([agg(base[k][0], S.VISIBLE) for k in FOLDS])), 4),
            cand_mean=round(float(np.mean([agg(cand[best][k][0], S.VISIBLE) for k in FOLDS])), 4),
            per_shift_gain={s: per_shift[s] for s in S.VISIBLE},
            standalone_visible_mean=sa,
            configs=[dict(misfit_weight=c[0], vote=c[1], reward=round(SS[c]['reward'], 5),
                          per_fold_gain=[round(x, 4) for x in SS[c]['vis']]) for c in configs],
            theta_probe=ridge_info, likelihood=lik_info),
        tail=dict(exc_base_pct=np.round(exc_b, 3).tolist(), exc_cand_pct=np.round(exc_c, 3).tolist(),
                  nominal_pct=[1, .5, .1], penalty=round(b['pen'], 5)),
        runtime=dict(sec_per_1k_maps=round(R.dev_s / max(R.nmaps, 1) * 1000, 3) if R.nmaps else None,
                     maps_this_run=R.nmaps, extract_s=round(t_extract, 1), fit_s=round(t_fit, 1)),
        hidden=dict(shifts=HIDDEN, hidden_ok=bool(np.mean(b['hid']) > 0),
                    per_fold_gain=[round(x, 4) for x in b['hid']], mean_gain=round(float(np.mean(b['hid'])), 4),
                    per_shift_gain={s: per_shift[s] for s in HIDDEN},
                    all19_base=round(float(np.mean([agg(base[k][0], S.SHIFTS) for k in FOLDS])), 4),
                    all19_cand=round(float(np.mean([agg(cand[best][k][0], S.SHIFTS) for k in FOLDS])), 4)))


def proposer_view(r):
    pv = {k: v for k, v in r.items() if k not in ('hidden', 'traceback')}
    pv['hidden_ok'] = (r.get('hidden') or {}).get('hidden_ok')
    return pv


# ================================================================================================ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('candidates', type=Path, nargs='+')
    ap.add_argument('--out', type=Path, required=True, help='output dir (one subdir per candidate if several)')
    ap.add_argument('--mode', default='rl', choices=['rl', 'screen', 'full'])
    ap.add_argument('--stack', default='v2')
    ap.add_argument('--chip', type=int, default=None)
    ap.add_argument('--budget', type=float, default=None, help='max TPU seconds per 10k maps (default per mode)')
    ap.add_argument('--timeout', type=float, default=None)
    args = ap.parse_args()
    init(args.chip, args.stack)
    for c in args.candidates:
        out = args.out / c.stem if len(args.candidates) > 1 else args.out
        out.mkdir(parents=True, exist_ok=True)
        src = c.read_text()
        (out / 'candidate.py').write_text(src)
        r = grade_source(src, args.mode, budget=args.budget, timeout=args.timeout)
        r['candidate'] = str(c)
        (out / 'result.json').write_text(json.dumps(r, indent=1))
        (out / 'proposer_view.json').write_text(json.dumps(proposer_view(r), indent=1))
        log(f'{c.name}: ok={r["ok"]} reward={r["reward"]} hidden_ok={(r.get("hidden") or {}).get("hidden_ok")} '
            f'all19={(r.get("hidden") or {}).get("all19_cand")} total={r["runtime"].get("total_s")}s '
            f'extract={r["runtime"].get("extract_s")}s fit={r["runtime"].get("fit_s")}s '
            f'fail={(r.get("contract") or {}).get("fail")}')


if __name__ == '__main__':
    main()
