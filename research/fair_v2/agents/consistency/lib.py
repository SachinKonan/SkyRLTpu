"""Shared helpers for the consistency agent (copied/adapted from fit_v3; shared files untouched)."""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))  # research/fair_v2 (model, groups, data)

from groups import groups  # noqa: E402
from model import phase2_score, Standardizer  # noqa: E402

ROOT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/v3')
OUT = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency')
SHIFTS15 = ['aniso0.7', 'bary6.5', 'bary9.0', 'baryDMO', 'boost0.15', 'gradient', 'mult+0.05', 'mult-0.05',
            'noise+0.05', 'noise-0.05', 'noise_t5', 'noisegrad0.15', 'peakcut0.3', 'smooth0.5', 'supp0.15']
NEW_SHIFTS = ['srcclust0.15', 'poisson120', 'ia0.2', 'psfpat']
# new coordinator shifts are included once both feature sources cover them
SHIFTS = SHIFTS15 + [s for s in NEW_SHIFTS
                     if all((ROOT / src / f'shift_{s}.npz').exists() for src in ('stats', 'cnn0'))]
SETS = ['calib', 'audit'] + ['shift_' + s for s in SHIFTS]

G, _ = groups()
PS = G['ps']
FAMILIES = {
    # stats-derived families (column indices into stats 466)
    'psL': ('stats', PS[:14]),
    'psS': ('stats', PS[14:]),
    'dir': ('stats', G['dir']),
    'onept': ('stats', np.concatenate([G['pdf'], G['moments']])),
    'extrema': ('stats', np.concatenate([G['peaks'], G['minima']])),
    'mf': ('stats', np.concatenate([G['v1'], G['v2']])),
    'dog': ('stats', G['dog']),
    'fine': ('stats', np.concatenate([G['s0'], G['s1']])),
    'coarse': ('stats', np.concatenate([G['s3'], G['s4']])),
    'cnn': ('cnn0', np.arange(261)),
    'cnn1': ('cnn1', np.arange(261)),
    # round-2 views: denoiser residual / stats on the denoised map / statistics-agent sources
    'den_r': ('den_r', np.arange(112)),
    'den_sd': ('den_s', G['dog']),
    'den_sa': ('den_s', np.arange(466)),
    'stx_bisp': ('stx_bisp', np.arange(92)),
    'stx_ipow': ('stx_ipow', np.arange(44)),
    'stx_nois': ('stx_nois', np.arange(54)),
    'stx_pkcl': ('stx_pkcl', np.arange(82)),
    'stx_prof': ('stx_prof', np.arange(200)),
    'stx_scat': ('stx_scat', np.arange(66)),
    'stx_tail': ('stx_tail', np.arange(45)),
    # statistics-agent members used by the coordinator (x = drop columns, k = keep columns)
    'N2': ('stx_nois', np.setdiff1d(np.arange(54), [6, 8, 26, 28, 53])),
    'PsBh': [('stats', PS), ('stx_bisp', np.arange(62, 92))],
    # four scale bands (PS quarter + one smoothing scale + one DoG band)
    'b0': ('stats', np.concatenate([PS[21:], G['s0'], G['dog'][0:14]])),
    'b1': ('stats', np.concatenate([PS[14:21], G['s1'], G['dog'][14:28]])),
    'b2': ('stats', np.concatenate([PS[7:14], G['s2'], G['dog'][28:42]])),
    'b3': ('stats', np.concatenate([PS[:7], G['s3'], G['s4'], G['dog'][42:]])),
    # clean scale split for the Dai-Seljak style scale-consistency score
    'small': ('stats', np.concatenate([PS[14:], G['s0'], G['s1'], G['dog'][:28]])),
    'mid': ('stats', G['s2']),
    'large': ('stats', np.concatenate([PS[:14], G['s3'], G['s4'], G['dog'][28:]])),
}


def load(src, name, key='feats'):
    with np.load(ROOT / src / f'{name}.npz') as f:
        return f[key]


def cols(fam, name):
    spec = FAMILIES[fam]
    if isinstance(spec, list):
        return np.concatenate([load(src, name)[..., c] for src, c in spec], -1)
    src, c = spec
    return load(src, name)[..., c]


def labels(name):
    return load('stats', name, 'labels')


def evaluate(za, zs):
    """za: audit normal scores; zs: dict shift-> scores. Returns dict."""
    q = np.quantile(za, [.5, .9, .99, .995, .999]).round(3).tolist()
    sh = {s: round(phase2_score(za, zs[s]), 4) for s in SHIFTS if s in zs}
    out = dict(nq=q, shift_mean=round(float(np.mean([sh[s] for s in SHIFTS15])), 4), shifts=sh)
    new = [s for s in NEW_SHIFTS if s in sh]
    if new:
        out['new_mean'] = round(float(np.mean([sh[s] for s in new])), 4)
    return out


def fmt(name, r, short=True):
    keys = ['aniso0.7', 'bary6.5', 'bary9.0', 'baryDMO', 'boost0.15', 'gradient', 'noisegrad0.15', 'peakcut0.3',
            'smooth0.5', 'supp0.15']
    s = f'{name:34s} mean={r["shift_mean"]:.4f} q99/99.5/99.9={r["nq"][2]:.2f}/{r["nq"][3]:.2f}/{r["nq"][4]:.2f}'
    if 'new_mean' in r:
        s += f' new4={r["new_mean"]:.3f}'
    if short:
        s += ' | ' + ' '.join(f'{k[:6]}={r["shifts"][k]:.3f}' for k in keys)
    return s


def r1_z(tag):
    with np.load(ROOT / 'runs' / 'r1' / f'z_{tag}.npz') as f:
        return {k: f[k] for k in f.files}
