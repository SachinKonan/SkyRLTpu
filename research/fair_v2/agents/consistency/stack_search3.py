import itertools, json, sys
import numpy as np
from lco3_analyze import components, evaluate, combo
comps = {k: components(k, 'std') for k in (0, 1, 2)}
print('members available:', sorted(comps[0]))
def ev(w):
    rs = [evaluate(combo(comps[k], w)) for k in (0, 1, 2)]
    return np.array([r['m19'] for r in rs]), np.mean([r['exc'] for r in rs], 0)
v304, _ = ev({'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1})
SD = {'Sd': .5, 'C': 1, 'Dr': 2.5, 'xSL': 1, 'consP8': .5, 'consR': 1.5, 'N2': .5, 'PsBh': .5}
sd, _ = ev(SD)
print('v3-04', v304.round(3), round(v304.mean(), 4), '| STACK_D', sd.round(3), round(sd.mean(), 4))
extra = [m for m in sys.argv[1].split(',') if m in comps[0]]
grid = [0, 0.5, 1, 2]
rows = []
for ws in itertools.product(grid, repeat=len(extra)):
    w = dict(SD); w.update({m: x for m, x in zip(extra, ws) if x})
    s, exc = ev(w)
    rows.append((float(s.mean()), float((s - sd).min()), s.round(3).tolist(), exc.round(2).tolist(), {m: x for m, x in zip(extra, ws)}))
for r in sorted(rows, key=lambda r: -r[0])[:12]:
    print(f'  mean {r[0]:.4f} worst-vs-STACK_D {r[1]:+.4f} sets {r[2]} exc {r[3]} add {r[4]}')
