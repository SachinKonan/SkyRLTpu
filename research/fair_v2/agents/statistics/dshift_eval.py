"""Score the private BCM-like diagnostic shifts with the frozen r2 models (same pickles as the
fit_eval runs) and report phase-2 metric vs the audit normal z of each model / mean combo."""
import pickle
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from groups import groups  # noqa: E402
from model import phase2_score  # noqa: E402

RUN = Path(sys.argv[1]); TAGS = sys.argv[2].split(','); COMBOS = sys.argv[3].split(',') if len(sys.argv) > 3 else []
DS = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/statistics/dshift')
NAMES = sorted(p.stem for p in (DS / 'stats').glob('shift_*.npz'))
G, _ = groups()


def columns(spec, name):
    parts = []
    for item in spec.split('|'):
        src, subset = item.split(':')
        f = np.load(DS / src / f'{name}.npz')['feats']
        if subset == 'raw':
            cols = np.arange(f.shape[-1])
        elif subset[0] in 'xk' and subset[1:2].isdigit():
            sel = []
            for tok in subset[1:].split('.'):
                a, _, b = tok.partition('-'); sel += list(range(int(a), int(b or a) + 1))
            cols = np.setdiff1d(np.arange(f.shape[-1]), sel) if subset[0] == 'x' else np.array(sel)
        else:
            cols = np.concatenate([G[g] for g in subset.split('+')])
        parts.append(f[..., cols])
    return np.concatenate(parts, -1)


zs = {}
for tag in TAGS:
    d = pickle.load(open(RUN / f'model_{tag}.pkl', 'rb'))
    lk, std, spec = d['lk'], d['std'], d['spec']
    za = np.load(RUN / f'z_{tag}.npz')['audit']
    zs[tag] = {'audit': za}
    for n in NAMES:
        f = columns(spec, n)
        _, th = lk.profile(f.reshape(-1, f.shape[-1]))
        zs[tag][n] = std(lk.last_copula, th)
    print(f'{tag:6s} ' + ' '.join(f'{n[6:]}={phase2_score(za, zs[tag][n]):.3f}' for n in NAMES), flush=True)
for combo in filter(None, COMBOS):
    mode, _, body = combo.rpartition(':'); parts = body.split('+')
    comb = lambda n: (np.max if mode == 'max' else np.mean)(np.stack([zs[p][n] for p in parts]), 0)
    za = comb('audit')
    print(f'{combo:24s} ' + ' '.join(f'{n[6:]}={phase2_score(za, comb(n)):.3f}' for n in NAMES), flush=True)
