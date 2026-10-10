"""Add field-level z (FA = difa profile score) to the current LB-best recipe REC1; paired bootstrap; export test z.

REC1 = (z_Sd + z_C + 0.5 z_xSL + 0.5 z_consP8)/3 (consistency agent export). Mixed = (3*REC1 + w*z_FA)/(3+w).
Paired bootstrap resamples audit3 maps (shifts are built from the same audit3 maps, same order).
"""
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model import phase2_score  # noqa: E402

E = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency/export')
O = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/diffusion')
R = dict(np.load(E / 'z_BASE+0.5xSL+0.5consP8.npz'))
R['test'] = np.load(E / 'test_z_BASE+0.5xSL+0.5consP8.npy')
FA = dict(np.load(O / 'fit' / 'z_FA.npz'))
RR = dict(np.load(O / 'fit_dst' / 'z_R.npz'))
sh = sorted(k for k in R if k.startswith('shift_'))
keys = ['audit', 'test'] + sh


def mix(parts):
    t = sum(w for _, w in parts)
    return {k: sum(z[k] * w for z, w in parts) / t for k in keys}


def m19(z, idx=None):
    a = z['audit'] if idx is None else z['audit'][idx]
    return np.mean([phase2_score(a, z[s] if idx is None else z[s][idx]) for s in sh])


cands = {'REC1+0.5FA': mix([(R, 3.0), (FA, 0.5)]), 'REC1+1FA+0.5R': mix([(R, 3.0), (FA, 1.0), (RR, 0.5)])}
rng = np.random.default_rng(0)
n = len(R['audit'])
for name, Z in cands.items():
    d = []
    for b in range(100):
        idx = rng.integers(0, n, n)
        d.append(m19(Z, idx) - m19(R, idx))
    d = np.array(d)
    print(f'{name}: m19 {m19(Z):.4f} vs REC1 {m19(R):.4f}; boot diff {d.mean():+.4f} +- {d.std():.4f}, P>0 {np.mean(d > 0):.2f}',
          flush=True)
    t = Z['test']
    assert t.shape == (10000,) and np.isfinite(t).all()
    np.save(O / f'test_z_{name}.npy', t.astype(np.float64))
np.save(O / 'test_z_FA.npy', FA['test'].astype(np.float64))
print('exported', flush=True)
