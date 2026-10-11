"""Cache per-fold member z's (v3-16b members) + cosmology ids of every row, for the lever-1 combiner study."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'agents' / 'consistency'))
import grade as G  # noqa: E402
import data as D  # noqa: E402
import lib  # noqa: E402
from holdout3 import masks  # noqa: E402
from stack_levers import DATA, load_fold  # noqa: E402
from ol_lco import fold_member, ol_raw  # noqa: E402

MEMBERS = ['Sd', 'Dr', 'xSL', 'consP8', 'consRplus', 'N2', 'PsBh', 'Oh', 'Xdp', 'Ol']
OUT = DATA / 'lever1_folds.npz'


def main():
    views = json.loads((DATA / 'cache_v2' / 'stack.json').read_text())['views']
    ci, ai = D.index('calib3'), D.index('audit3')
    arrs = {}
    for k in G.FOLDS:
        comp, TH = load_fold(k)
        comp['consRplus'] = G.cons(TH, views, G.NAMES)
        comp['Ol'] = fold_member(k, ol_raw)
        m = masks(k)
        cos_out = np.concatenate([np.repeat(ci[m['calib'], 0], 2), ai[m['audit'], 0]])
        cos_sh = ai[m['audit'], 0]
        for mem in MEMBERS:
            for n in ['out'] + ['sh_' + s for s in lib.SHIFTS]:
                arrs[f'{k}/{mem}/{n}'] = np.asarray(comp[mem][n], np.float64)
        assert len(cos_out) == len(arrs[f'{k}/Sd/out']), (len(cos_out), len(arrs[f'{k}/Sd/out']))
        assert len(cos_sh) == len(arrs[f'{k}/Sd/sh_psfpat']), (len(cos_sh), len(arrs[f'{k}/Sd/sh_psfpat']))
        arrs[f'{k}/cos_out'] = cos_out
        arrs[f'{k}/cos_sh'] = cos_sh
        print('fold', k, 'out', len(cos_out), 'sh rows', len(cos_sh), 'cosmologies', len(np.unique(cos_out)), flush=True)
    np.savez(OUT, **arrs)
    print('wrote', OUT)


if __name__ == '__main__':
    main()
