"""Stacking search on the LCO proxy: keep v3-04's members, add robust members; rank recipes by the
worst-set improvement over v3-04 (robust), then by mean."""
import itertools, json
import numpy as np
from lco3_analyze import components, evaluate, combo
comps = {k: components(k, 'std') for k in (0, 1, 2)}
def ev(w):
    rs = [evaluate(combo(comps[k], w)) for k in (0, 1, 2)]
    return np.array([r['m19'] for r in rs]), np.mean([r['exc'] for r in rs], 0)
base, _ = ev({'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1})
rows = []
for sd, c, dr, cp8, cr, n2, pb, xs in itertools.product([0.5, 1], [0.25, 0.5, 0.75, 1], [1, 1.5, 2], [0.5, 1],
                                                         [1, 2, 3], [0.5, 1, 1.5], [0, 0.5], [1, 1.5]):
    w = {'Sd': sd, 'C': c, 'Dr': dr, 'xSL': xs, 'consP8': cp8, 'consR': cr, 'N2': n2, 'PsBh': pb}
    w = {k: v for k, v in w.items() if v}
    s, exc = ev(w)
    rows.append((float((s - base).min()), float(s.mean()), s.round(3).tolist(), exc.round(2).tolist(), w))
rows.sort(key=lambda r: (-r[0], -r[1]))
print('v3-04 sets', base.round(3).tolist(), 'mean %.4f' % base.mean())
print('top by worst-set gain:')
for r in rows[:12]:
    print(f'  worst+{r[0]:.4f} mean {r[1]:.4f} sets {r[2]} exc {r[3]} {r[4]}')
rows.sort(key=lambda r: -r[1])
print('top by mean (with worst-set gain):')
for r in rows[:8]:
    print(f'  mean {r[1]:.4f} worst+{r[0]:.4f} sets {r[2]} exc {r[3]} {r[4]}')
json.dump(rows[:200], open('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency/stack_search.json', 'w'))
