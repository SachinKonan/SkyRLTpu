"""Add a factory family to the v3-12 recipe and export submission candidates (CPU).

  python factory/export_family.py --family dirpower --run <factory grade run dir> [--export]

Full-data parts mirror how v3-12 was built (consistency2/c2export.py + probes.py export, merge.py, export.py):
- existing members from consistency/export4/components_STACKDp3_denb.npz, SS from diffusion2/z/z_SS.npz and Oh
  from outlier2/runs/r1/z_Oh.npz, combined as a weighted mean of calib3-standardized z (merge.py);
- the family's misfit member: consistency/fam/<family>.npz (factory/fit_family.py == fit_families.py) through
  analyze.standardize(cop, th), as export3.components does for likelihood members;
- the family's vote: probes.full_probe ridge theta-probe (fit on fit3, alpha on calib3), added as an extra view
  to consRplus (views RB + pden_b + p<family>), Mahalanobis covariance on calib3 (export3.cons).
Test maps are only scored with these frozen models. LCO numbers come from the factory fold caches (grade.py).
"""
import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent / 'agents' / 'consistency'))
import grade as G  # noqa: E402
import lib  # noqa: E402
import probes  # noqa: E402
from analyze import ev, fam, standardize  # noqa: E402
from export3 import cons  # noqa: E402
from lco3_analyze import components as lco_components  # noqa: E402

DATA = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data')
EXPORT4 = DATA / 'agents/consistency/export4/components_STACKDp3_denb.npz'
PDIR = DATA / 'agents/consistency2/probes'
SS_Z = DATA / 'agents/diffusion2/z/z_SS.npz'
OH_Z = DATA / 'agents/outlier2/runs/r1/z_Oh.npz'
V312 = {'Sd': .5, 'C': 1, 'Dr': 2.5, 'xSL': 1, 'consP8': .5, 'consRplus': 3, 'N2': .5, 'PsBh': .5, 'SS': 1, 'Oh': 1.5}


def recipes():
    no_oh = {k: v for k, v in V312.items() if k != 'Oh'}
    return {
        'v3-12': (V312, False),
        'v3-12-noOh': (no_oh, False),
        'v3-13a': ({**V312, 'X': 2}, True),
        'v3-13b': ({**no_oh, 'X': 2}, True),
        'v3-13c': ({**V312, 'X': 1}, True),
    }


def combine(comp, w, keys):
    W = sum(w.values())
    return {k: sum(wi * comp[m][k] for m, wi in w.items()) / W for k in keys}


def lco_table(run):
    """All-19 / visible / hidden LCO per recipe from the factory fold caches."""
    rows = {}
    for k in G.FOLDS:
        C = G.build_cache(k)
        with open(run / f'fold{k}_full.pkl', 'rb') as fh:
            P = pickle.load(fh)
        take = P['take']
        comp = {m: G.sub(z, take) for m, z in C['comp'].items()}
        comp['Oh'] = G.sub(lco_components(k, 'std')['Oh'], take)
        TH = {f: G.sub(v, take) for f, v in C['TH'].items()}
        comp['consRplus'] = cons_names(TH, G.VIEWS0)
        TH['cand'] = P['ridge']
        cons_x = cons_names(TH, G.VIEWS0 + ['cand'])
        comp['X'] = P['mis']
        for tag, (w, with_vote) in recipes().items():
            c = dict(comp)
            if with_vote:
                c['consRplus'] = cons_x
            W = sum(w.values())
            z = {n: sum(wi * c[m][n] for m, wi in w.items()) / W for n in G.NAMES}
            sh, exc = G.metrics(z)
            rows.setdefault(tag, []).append((sh, exc))
    out = {}
    for tag, per in rows.items():
        agg = lambda which: float(np.mean([np.mean([sh[s] for s in which]) for sh, _ in per]))
        out[tag] = dict(all19=round(agg(lib.SHIFTS), 4), visible=round(agg(G.VISIBLE), 4),
                        hidden=round(agg(G.HIDDEN), 4),
                        per_fold_all19=[round(float(np.mean([sh[s] for s in lib.SHIFTS])), 4) for sh, _ in per],
                        exc_pct=np.round(np.mean([e for _, e in per], 0), 3).tolist())
    return out


def cons_names(TH, views):
    from lco3_analyze import cons as lco_cons
    return lco_cons(TH, views, G.NAMES)


def full_components(family):
    sets = lib.SETS + ['test']
    E = np.load(EXPORT4)
    comp = {m: {s: E[f'{m}/{s}'] for s in sets} for m in V312 if m not in ('SS', 'Oh')}
    with np.load(SS_Z) as d:
        comp['SS'] = {s: d[s] for s in sets}
    with np.load(OH_Z) as d:
        comp['Oh'] = {s: d[s] for s in sets}
    # consRplus views (c2export.py): consR bands + den_b probe, then + the family's probe
    TH = {}
    for f in [x for x in probes.RB if not x.startswith('pred')]:
        F = fam(f)
        TH[f] = {s: F[f'{s}/th'] for s in sets}
    TH['pred0'] = {s: lib.load('cnn0', s)[..., -5:].reshape(-1, 5) for s in sets}
    TH['pred1'] = {s: lib.load('cnn1', s)[..., -5:].reshape(-1, 5) for s in sets}
    with np.load(PDIR / 'den_b_full.npz') as H:
        TH['pden_b'] = {s: H[s] for s in sets}
    rebuilt = cons(TH, probes.RB + ['pden_b'], sets)
    dev = max(float(np.abs(rebuilt[s] - comp['consRplus'][s]).max()) for s in sets)
    probes.PROBES[family] = (family, slice(None))
    TH['p' + family], alpha, mse = probes.full_probe(family)
    comp['consRplus_x'] = cons(TH, probes.RB + ['pden_b', 'p' + family], sets)
    F = fam(family)
    comp['X'] = standardize({s: F[f'{s}/cop'] for s in sets}, {s: F[f'{s}/th'] for s in sets})
    info = dict(consRplus_rebuild_max_abs_dev=dev, probe_alpha=alpha, probe_calib_mse=round(float(mse), 4))
    return comp, info


def rank_moves(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    rho = float(np.corrcoef(ra, rb)[0, 1])
    return round(rho, 5), round(float(np.mean(np.abs(ra - rb) > 0.05 * len(a))), 4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--family', required=True)
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--export', action='store_true')
    args = ap.parse_args()
    report = dict(lco=lco_table(args.run))
    print(json.dumps(report['lco'], indent=1), flush=True)
    comp, info = full_components(args.family)
    report['full'] = info
    print(json.dumps(info), flush=True)
    sets = lib.SETS + ['test']
    ref = np.load(DATA / 'merged' / 'v3-12.npy')
    tests = {}
    for tag, (w, with_vote) in recipes().items():
        c = dict(comp)
        if with_vote:
            c['consRplus'] = comp['consRplus_x']
        z = combine(c, w, sets)
        r = ev(z)
        t = z['test']
        assert t.shape == (10000,) and np.isfinite(t).all()
        tests[tag] = t
        rho, moved = rank_moves(ref, t)
        report.setdefault('recipes', {})[tag] = dict(
            weights=w, family_vote_in_consRplus=with_vote, audit_m15=r['shift_mean'], audit_new4=r.get('new_mean'),
            audit_q99_99p5_99p9=[r['nq'][2], r['nq'][3], r['nq'][4]], test_rank_corr_vs_v3_12=rho,
            test_frac_rank_moved_gt5pct=moved)
        print(tag, json.dumps(report['recipes'][tag]), flush=True)
    assert np.abs(tests['v3-12'] - ref).max() == 0, 'v3-12 not reproduced'
    out = DATA / 'factory' / f'export_{args.family}'
    out.mkdir(parents=True, exist_ok=True)
    (out / 'report.json').write_text(json.dumps(report, indent=1))
    if args.export:
        for tag in ('v3-13a', 'v3-13b', 'v3-13c'):
            npy = DATA / 'merged' / f'{tag}.npy'
            np.save(npy, tests[tag].astype(np.float64))
            name = {'v3-13a': 'v3-13a-v312+dirpower2', 'v3-13b': 'v3-13b-v312noOh+dirpower2',
                    'v3-13c': 'v3-13c-v312+dirpower1'}[tag]
            subprocess.run([sys.executable, str(HERE.parent / 'export.py'), str(npy),
                            str(DATA / 'submissions' / f'{name}.zip')], check=True)


if __name__ == '__main__':
    main()
