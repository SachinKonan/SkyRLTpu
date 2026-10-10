import itertools, json
import numpy as np
from lco3_analyze import components, evaluate, combo
comps = {k: components(k, 'std') for k in (0, 1, 2)}
def ev(w):
    rs = [evaluate(combo(comps[k], w)) for k in (0, 1, 2)]
    return np.array([r['m19'] for r in rs]), np.mean([r['exc'] for r in rs], 0)
base, _ = ev({'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1})
rows = []
for sd, c, dr, cr, n2 in itertools.product([0.5, 1], [0.5, 0.75, 1], [2, 2.5, 3, 4], [1, 1.5, 2], [0.5, 1, 1.5]):
    w = {'Sd': sd, 'C': c, 'Dr': dr, 'xSL': 1, 'consP8': 0.5, 'consR': cr, 'N2': n2, 'PsBh': 0.5}
    s, exc = ev(w)
    rows.append((float((s - base).min()), float(s.mean()), s.round(3).tolist(), exc.round(2).tolist(), w))
print('v3-04', base.round(3).tolist(), round(float(base.mean()), 4))
print('dominating (worst-set gain >= 0), best mean:')
for r in sorted([r for r in rows if r[0] >= 0], key=lambda r: -r[1])[:8]:
    print(f'  mean {r[1]:.4f} worst+{r[0]:.4f} sets {r[2]} exc {r[3]} {r[4]}')
print('overall best mean:')
for r in sorted(rows, key=lambda r: -r[1])[:5]:
    print(f'  mean {r[1]:.4f} worst+{r[0]:.4f} sets {r[2]} exc {r[3]} {r[4]}')
