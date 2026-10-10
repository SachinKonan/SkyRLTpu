import numpy as np
from lco3_analyze import components, evaluate, combo
comps = {k: components(k, 'std') for k in (0, 1, 2)}
def ev(w):
    rs = [evaluate(combo(comps[k], w)) for k in (0, 1, 2)]
    s = np.array([r['m19'] for r in rs]); return f"mean {s.mean():.4f} sets {s.round(3).tolist()}"
SD = {'Sd': .5, 'C': 1, 'Dr': 2.5, 'xSL': 1, 'consP8': .5, 'consR': 1.5, 'N2': .5, 'PsBh': .5}
for name, w in {'Dr alone': {'Dr': 1}, 'Dw alone': {'Dw': 1}, 'STACK_D': SD,
                'STACK_D Dr->Dw': {**{k: v for k, v in SD.items() if k != 'Dr'}, 'Dw': 2.5},
                'STACK_D + Dw 1': {**SD, 'Dw': 1}}.items():
    print(f'{name:18s} {ev(w)}', flush=True)
