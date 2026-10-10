"""Evaluate hand-picked recipes under LCO (all held-out sets) for a given variant."""
import sys
import json
import numpy as np
import lib
from lco3_analyze import components, evaluate, combo

R = {
    'v3-04': {'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1},
    'R_A': {'Sd': .5, 'C': .5, 'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consP8': .5, 'consR': 2},
    'R_B': {'Dr': 1, 'N2': 2, 'xSL': .5, 'consR': 2},
    'R_C': {'Dr': 1, 'N2': 1, 'K': .5, 'PsBh': .5, 'xSL': .5, 'consR': 2},
    'R_D': {'Sd': .5, 'C': .5, 'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consR': 2},
    'R_E': {'Dr': 1, 'N2': 1, 'xSL': 1, 'consR': 2},
    'R_F': {'Dr': 1, 'N2': 1, 'xSL': 1, 'consR': 1, 'consP8': .5, 'C': .5},
    'R_G': {'Sd': .5, 'Dr': 1, 'N2': 1, 'xSL': 1, 'consR': 2},
    'R_H': {'Dr': 1.5, 'N2': 1.5, 'PsBh': .5, 'xSL': 1, 'consR': 2, 'C': .5},
}
variant = sys.argv[1] if len(sys.argv) > 1 else 'std'
extra = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
R.update(extra)
comps = {k: components(k, variant) for k in (0, 1, 2)}
for name, w in R.items():
    if not all(c in comps[0] for c in w):
        continue
    rs = [evaluate(combo(comps[k], w)) for k in (0, 1, 2)]
    exc = np.mean([r['exc'] for r in rs], 0)
    m19 = np.mean([r['m19'] for r in rs]); m15 = np.mean([r['m15'] for r in rs])
    keys = ['bary6.5', 'baryDMO', 'bary9.0', 'supp0.15', 'ia0.2', 'psfpat']
    print(f'{variant:4s} {name:7s} LCO m19={m19:.4f} sets=' + '/'.join(f'{r["m19"]:.3f}' for r in rs) +
          f' m15={m15:.4f} exceed={exc[0]:.2f}/{exc[1]:.2f}/{exc[2]:.2f}% | ' +
          ' '.join(f'{s[:6]}={np.mean([r["sh"][s] for r in rs]):.3f}' for s in keys) + f'  {w}', flush=True)
