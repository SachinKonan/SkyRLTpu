import numpy as np
from lco3_analyze import components, evaluate, combo
comps = {k: components(k, 'std') for k in (0, 1, 2)}
def ev(w):
    rs = [evaluate(combo(comps[k], w)) for k in (0, 1, 2)]
    s = np.array([r['m19'] for r in rs]); exc = np.mean([r['exc'] for r in rs], 0)
    return f"mean {s.mean():.4f} sets {s.round(3).tolist()} exc {exc.round(2).tolist()}"
SD = {'Sd': .5, 'C': 1, 'Dr': 2.5, 'xSL': 1, 'consP8': .5, 'consR': 1.5, 'N2': .5, 'PsBh': .5}
R = {'Dr alone': {'Dr': 1}, 'De alone': {'De': 1}, 'Dr+De': {'Dr': 1, 'De': 1},
     'STACK_D': SD,
     'STACK_D Dr->De': {**{k: v for k, v in SD.items() if k != 'Dr'}, 'De': 2.5},
     'STACK_D + De 1.25 (Dr 1.25)': {**SD, 'Dr': 1.25, 'De': 1.25},
     'STACK_D + De 2.5': {**SD, 'De': 2.5},
     'STACK_D Dr->De + SS 1': {**{k: v for k, v in SD.items() if k != 'Dr'}, 'De': 2.5, 'SS': 1},
     'STACK_D Dr->De + SS 1 + Oh 1': {**{k: v for k, v in SD.items() if k != 'Dr'}, 'De': 2.5, 'SS': 1, 'Oh': 1},
     'STACK_D Dr->De 3.5 + SS 1': {**{k: v for k, v in SD.items() if k != 'Dr'}, 'De': 3.5, 'SS': 1},
     }
for name, w in R.items():
    print(f'{name:34s} {ev(w)}', flush=True)
