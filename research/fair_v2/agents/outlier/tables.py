"""Collect shift and family tables for REPORT.md from the run outputs."""
import json, sys
from pathlib import Path
O = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/outlier')
SH = ['mult+0.05', 'mult-0.05', 'smooth0.5', 'aniso0.7', 'noise+0.05', 'noise-0.05', 'noisegrad0.15', 'noise_t5',
      'supp0.15', 'boost0.15', 'peakcut0.3', 'gradient', 'bary9.0', 'bary6.5', 'baryDMO', 'srcclust0.15',
      'poisson120', 'ia0.2', 'psfpat']
HO_OUT = {'peakcut0.3', 'bary9.0', 'bary6.5', 'baryDMO', 'srcclust0.15', 'poisson120', 'ia0.2', 'psfpat'}
cols = []
def add(name, path, key):
    p = O / path
    if not p.exists():
        return
    r = json.loads(p.read_text())
    if key not in r:
        return
    cols.append((name, r[key]))
add('base Sd+C', 'fit_all/report.json', 'combo:mean:Sd+C')
for run in sys.argv[1:]:
    tag, fit, score = run.split(':')
    add(f'{tag} L', f'{score}/score_report.json', 'L')
    add(f'{tag} O', f'{fit}/report.json', 'O')
    add(f'{tag} O+C+Sd', f'{fit}/report.json', 'combo:mean:O+C+Sd')
    add(f'{tag} L+Sd+C', f'{score}/score_report.json', 'combo:mean:L+Sd+C')
    add(f'{tag} L+Sd', f'{score}/score_report.json', 'combo:mean:L+Sd')
    add(f'{tag} L+O+Sd+C', f'{score}/score_report.json', 'combo:mean:L+O+Sd+C')
print('| shift | HO status | ' + ' | '.join(c[0] for c in cols) + ' |')
print('|---|---|' + '---|' * len(cols))
for s in SH:
    print(f'| {s} | {"held-out" if s in HO_OUT else "in-family"} | ' + ' | '.join(f'{c[1]["shifts"][s]:.3f}' for c in cols) + ' |')
print('| **shift_mean (19)** | | ' + ' | '.join(f'**{c[1]["shift_mean"]:.3f}**' for c in cols) + ' |')
ho = sorted(HO_OUT)
print('| mean over HO-held-out 8 | | ' + ' | '.join(f'{sum(c[1]["shifts"][s] for s in ho)/len(ho):.3f}' for c in cols) + ' |')
print('| normal 99% / 99.9% (audit3) | | ' + ' | '.join(f'{c[1]["normal_quantiles"][3]:.2f} / {c[1]["normal_quantiles"][5]:.2f}' for c in cols) + ' |')
