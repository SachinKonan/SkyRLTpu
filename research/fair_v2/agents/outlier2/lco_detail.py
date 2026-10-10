"""Per-shift LCO deltas and member correlations for OE recipes (CPU)."""
import numpy as np
import sys
sys.path.insert(0, "../consistency"); sys.path.insert(0, "../..")
import lib
from lco_analyze_oe import comps, V4, BIG
from lco3_analyze import evaluate, combo
cs = {k: comps(k) for k in (0, 1, 2)}
R = {'v3-04': V4, '+O': {**V4, 'O': 1}, '+Oh': {**V4, 'Oh': 1}, '+Oho': {**V4, 'Oho': 1}, '+0.5Oho': {**V4, 'Oho': .5},
     '+L': {**V4, 'L': 1}, '+Lho': {**V4, 'Lho': 1}, 'BIG': BIG, 'BIG+Oho': {**BIG, 'Oho': 1}, 'BIG+0.5Oho': {**BIG, 'Oho': .5},
     'BIG+Oh': {**BIG, 'Oh': 1}, 'BIG+0.5Oh': {**BIG, 'Oh': .5}, 'BIG+O': {**BIG, 'O': 1}}
res = {}
for n, w in R.items():
    rs = [evaluate(combo(cs[k], w)) for k in (0, 1, 2)]
    res[n] = dict(m19=np.mean([r['m19'] for r in rs]), exc=np.mean([r['exc'] for r in rs], 0),
                  sh={s: np.mean([r['sh'][s] for r in rs]) for s in lib.SHIFTS})
print('recipe      m19    exc99/99.5/99.9   ' + ' '.join(s[:7].rjust(7) for s in lib.SHIFTS))
for n, r in res.items():
    print(f'{n:10s} {r["m19"]:.4f} {r["exc"][0]:.2f}/{r["exc"][1]:.2f}/{r["exc"][2]:.2f}  ' + ' '.join(f'{r["sh"][s]:7.3f}' for s in lib.SHIFTS))
HO = ['peakcut0.3', 'ia0.2', 'poisson120']
NEAR = ['bary6.5', 'bary9.0', 'baryDMO']
for n in R:
    r = res[n]
    print(f'{n:10s} truly-held-out(peakcut,ia,poisson)={np.mean([r["sh"][s] for s in HO]):.4f}  bary3={np.mean([r["sh"][s] for s in NEAR]):.4f}')
# correlations of members on LCO held-out normal maps ('out'), set-averaged
mem = ['Sd', 'C', 'Dr', 'xSL', 'consP8', 'O', 'Oh', 'Oho', 'L', 'Lho']
C = np.mean([np.corrcoef(np.stack([cs[k][m]['out'] for m in mem])) for k in (0, 1, 2)], 0)
print('corr on LCO normal maps (spearman-free pearson):')
print('        ' + ' '.join(m.rjust(6) for m in mem))
for i, m in enumerate(mem):
    print(f'{m:7s} ' + ' '.join(f'{C[i, j]:6.2f}' for j in range(len(mem))))
