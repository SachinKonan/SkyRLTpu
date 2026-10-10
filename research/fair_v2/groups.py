"""Index groups of the stats.py feature vector (466 dims)."""
import numpy as np

NPS, NDIR, NSCALE = 28, 7, 5
BLOCK = dict(pdf=18, peaks=14, minima=14, v1=12, v2=12, moments=5)
NDOG, DOGB = 4, 14


def groups():
    g = {'ps': list(range(0, NPS)), 'dir': list(range(NPS, NPS + NDIR))}
    k = NPS + NDIR
    for i in range(NSCALE):
        for name, n in BLOCK.items():
            g.setdefault(name, []).extend(range(k, k + n))
            g.setdefault(f's{i}', []).extend(range(k, k + n))
            k += n
    g['dog'] = list(range(k, k + NDOG * DOGB))
    k += NDOG * DOGB
    g['all'] = list(range(k))
    return {name: np.array(v) for name, v in g.items()}, k
