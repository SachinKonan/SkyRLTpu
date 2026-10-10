"""consistency2 stage 2: consRplus (consRbands + probe theta-views) on the 3 LCO sets, and recipes.

All per-set quantities come from the LCO refits (lco3/set{k}_std, probes refit on fit3 minus the held-out
cosmologies); the consRplus Mahalanobis covariance is each set's kept-calib3 covariance.
"""
import argparse
import itertools
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'consistency'))
import lib  # noqa: E402
from analyze import mean_of  # noqa: E402
from lco3_analyze import components, evaluate, cons, pred_sets  # noqa: E402

O2 = Path('/scratch/gpfs/ZHUANGL/sk7524/fair-data/agents/consistency2')
PDIR = O2 / 'probes'
KS = (0, 1, 2)
RB = ['b0', 'b1', 'b2', 'b3', 'cnn', 'cnn1', 'pred0', 'pred1', 'dog', 'den_r', 'den_sd']
PROBES = ['den_b', 'cnn0e', 'cnn1e', 'dif', 'difa', 'difk', 'oe', 'oee', 'oe_head', 'difcat', 'oeho', 'oehoe', 'lat', 'denbdif']
HO_OUT = ['peakcut0.3', 'bary9.0', 'bary6.5', 'baryDMO', 'srcclust0.15', 'poisson120', 'ia0.2', 'psfpat']
LVIEWS = ['dif', 'difa', 'difk', 'oe', 'oe_head', 'den_b']


def load_set(k):
    cp = O2 / f'comps_set{k}.pkl'
    if cp.exists():
        with open(cp, 'rb') as f:
            return pickle.load(f)
    comp = components(k, 'std')
    d = lib.OUT / 'lco3' / f'set{k}_std'
    names = ['calib', 'in', 'out'] + ['sh_' + s for s in lib.SHIFTS]
    TH = {}
    for f in [x for x in RB if not x.startswith('pred')]:
        with np.load(d / f'{f}.npz') as H:
            TH[f] = {n: H[f'{n}/th'] for n in names}
    TH['pred0'] = pred_sets('cnn0', k)
    TH['pred1'] = pred_sets('cnn1', k)
    for f in LVIEWS:
        with np.load(d / f'{f}.npz') as H:
            TH['L' + f] = {n: H[f'{n}/th'] for n in names}
    for p in PROBES:
        with np.load(PDIR / f'{p}_set{k}.npz') as H:
            TH['p' + p] = {n: H[n] for n in names}
    res = (comp, TH, names)
    with open(cp, 'wb') as f:
        pickle.dump(res, f)
    return res


S = {}


def get(k):
    if k not in S:
        S[k] = load_set(k)
    return S[k]


def comp_with(k, views):
    comp, TH, names = get(k)
    c = dict(comp)
    if views is not None:
        c['consRplus'] = cons(TH, views, names)
    return c


def lco(w, views=None, cache={}):
    rs = []
    for k in KS:
        key = (k, tuple(views) if views else None)
        if key not in cache:
            cache[key] = comp_with(k, views)
        c = cache[key]
        rs.append(evaluate(mean_of(*[c[x] for x in w], w=list(w.values()))))
    exc = np.mean([r['exc'] for r in rs], 0)
    sh = {s: float(np.mean([r['sh'][s] for r in rs])) for s in lib.SHIFTS}
    return dict(m=float(np.mean([r['m19'] for r in rs])), sets=[r['m19'] for r in rs], exc=exc.tolist(),
                m15=float(np.mean([r['m15'] for r in rs])), sh=sh)


def line(name, r):
    keys = ['bary6.5', 'baryDMO', 'supp0.15', 'smooth0.5', 'ia0.2', 'srcclust0.15', 'psfpat']
    return (f'{name:44s} LCO={r["m"]:.4f} sets={" ".join(f"{x:.3f}" for x in r["sets"])} '
            f'exc={r["exc"][0]:.2f}/{r["exc"][1]:.2f}/{r["exc"][2]:.2f}% m15={r["m15"]:.4f} ho8={np.mean([r["sh"][s] for s in HO_OUT]):.4f} | '
            + ' '.join(f'{s[:6]}={r["sh"][s]:.3f}' for s in keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--stage', default='views')
    ap.add_argument('--views', default='')
    args = ap.parse_args()
    t0 = time.time()
    for k in KS:
        get(k)
    print(f'loaded {time.time() - t0:.0f}s', flush=True)
    if args.stage == 'views':
        print('=== standalone consistency (LCO)')
        base = lco({'consR': 1})
        print(line('consR (11 views)', base), flush=True)
        print(line('consP8', lco({'consP8': 1})), flush=True)
        single = {}
        for v in ['p' + p for p in PROBES] + ['L' + f for f in LVIEWS]:
            r = lco({'consRplus': 1}, RB + [v])
            single[v] = r['m']
            print(line(f'consR + {v}', r), flush=True)
        # greedy forward selection over probe views (export-able: probes + Ldif/Ldifa/Ldifk)
        cand = ['p' + p for p in PROBES] + ['Ldif', 'Ldifa', 'Ldifk']
        chosen, cur = [], base['m']
        while True:
            best = None
            for v in cand:
                if v in chosen:
                    continue
                r = lco({'consRplus': 1}, RB + chosen + [v])
                if best is None or r['m'] > best[1]['m']:
                    best = (v, r)
            if best is None or best[1]['m'] < cur + 0.0015:
                break
            chosen.append(best[0]); cur = best[1]['m']
            print(line(f'greedy +{best[0]} -> {chosen}', best[1]), flush=True)
            if time.time() - t0 > 500:
                print('time budget', flush=True); break
        allp = RB + ['p' + p for p in ['den_b', 'cnn0e', 'cnn1e', 'difa', 'oe']]
        print(line('consR + 5 probes (den_b,cnn0e,cnn1e,difa,oe)', lco({'consRplus': 1}, allp)), flush=True)
        print(line('probes only (5) + pred0', lco({'consRplus': 1},
                                                   ['pred0'] + ['p' + p for p in ['den_b', 'cnn0e', 'cnn1e', 'difa', 'oe']])), flush=True)
        (O2 / 'views.json').write_text(json.dumps(dict(single=single, greedy=chosen)))
    elif args.stage == 'greedy':
        cand = args.views.split(',')
        base = lco({'consR': 1})
        print(line('consR', base), flush=True)
        for v in cand:
            print(line(f'consR + {v}', lco({'consRplus': 1}, RB + [v])), flush=True)
        chosen, cur = [], base['m']
        while True:
            best = None
            for v in cand:
                if v in chosen:
                    continue
                r = lco({'consRplus': 1}, RB + chosen + [v])
                if best is None or r['m'] > best[1]['m']:
                    best = (v, r)
            if best is None or best[1]['m'] < cur + 0.0015:
                break
            chosen.append(best[0]); cur = best[1]['m']
            print(line(f'greedy +{best[0]} -> {chosen}', best[1]), flush=True)
        print('GREEDY', ','.join(chosen))
    elif args.stage == 'stackd':
        views = RB + [v for v in args.views.split(',') if v]
        print('consRplus views:', views)
        base = np.array(lco({'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1})['sets'])
        SD = {'Sd': .5, 'C': 1, 'Dr': 2.5, 'xSL': 1, 'consP8': .5, 'consR': 1.5, 'N2': .5, 'PsBh': .5}
        print(line('STACK_D (consR)', lco(SD)), flush=True)
        SDp = {k: v for k, v in SD.items() if k != 'consR'}; SDp['consRplus'] = 1.5
        print(line('STACK_D consR->consRplus', lco(SDp, views)), flush=True)
        for w2 in (2, 3):
            print(line(f'STACK_D consRplus {w2}', lco({**SDp, 'consRplus': w2}, views)), flush=True)
        rows = []
        for sd, c, dr, cp8, crp, n2, pb, xs in itertools.product([.5, 1], [.5, .75, 1], [1.5, 2, 2.5], [.5, 1],
                                                                 [1.5, 2, 3], [.5, 1], [0, .5], [1]):
            w = {'Sd': sd, 'C': c, 'Dr': dr, 'xSL': xs, 'consP8': cp8, 'consRplus': crp, 'N2': n2, 'PsBh': pb}
            w = {a: b for a, b in w.items() if b}
            r = lco(w, views)
            rows.append((float((np.array(r['sets']) - base).min()), r['m'], r['sets'], r['exc'], w))
            if time.time() - t0 > 520:
                print('grid time budget', len(rows), flush=True); break
        rows.sort(key=lambda x: (-x[0], -x[1]))
        print('top by worst-set gain over v3-04:')
        for g, m, sets, exc, w in rows[:8]:
            print(f'  worst+{g:.4f} LCO={m:.4f} sets={np.round(sets, 3).tolist()} exc={np.round(exc, 2).tolist()} {w}')
        rows.sort(key=lambda x: -x[1])
        print('top by mean with worst-set gain >= 0:')
        for g, m, sets, exc, w in [x for x in rows if x[0] >= 0][:6]:
            print(f'  LCO={m:.4f} worst+{g:.4f} sets={np.round(sets, 3).tolist()} exc={np.round(exc, 2).tolist()} {w}')
    elif args.stage == 'recipes':
        views = RB + [v for v in args.views.split(',') if v]
        print('consRplus views:', views)
        rows = []

        def rep(name, w, vv=views):
            r = lco(w, vv)
            print(line(name, r), flush=True)
            rows.append(dict(name=name, w=w, r=r))
        print(line('consRplus alone', lco({'consRplus': 1}, views)))
        V304 = {'Sd': 1, 'C': 1, 'Dr': 1, 'xSL': 1, 'consP8': 1}
        rep('v3-04', V304)
        rep('BIG_RE2', {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consR': 2})
        for wc in (1.5, 2, 3):
            rep(f'BIG+ consRplus {wc}', {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consRplus': wc})
        rep('BIG+ consRplus 1 + consR 1', {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consRplus': 1, 'consR': 1})
        rep('HEDGE_RE3 (BIG_RE2 +C.25 Sd.25)', {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consR': 2, 'C': .25, 'Sd': .25})
        rep('HEDGE+ (consRplus)', {'Dr': 1, 'N2': 1, 'PsBh': .5, 'xSL': 1, 'consRplus': 2, 'C': .25, 'Sd': .25})
        for extra_name, extra in [('+consR1', {'consR': 1}), ('+consR2', {'consR': 2}),
                                  ('+consRplus1', {'consRplus': 1}), ('+consRplus2', {'consRplus': 2}),
                                  ('+consRplus2 N2', {'consRplus': 2, 'N2': 1}),
                                  ('+consRplus2 N2 PsBh.5', {'consRplus': 2, 'N2': 1, 'PsBh': .5}),
                                  ('+consRplus3 N2 PsBh.5', {'consRplus': 3, 'N2': 1, 'PsBh': .5}),
                                  ('+consR2 N2 PsBh.5', {'consR': 2, 'N2': 1, 'PsBh': .5})]:
            rep('v3-04 ' + extra_name, {**V304, **extra})
        for cw in (.25, .5):
            for extra_name, extra in [('+consRplus2 N2 PsBh.5', {'consRplus': 2, 'N2': 1, 'PsBh': .5}),
                                      ('+consRplus3 N2 PsBh.5', {'consRplus': 3, 'N2': 1, 'PsBh': .5}),
                                      ('+consR2 N2 PsBh.5', {'consR': 2, 'N2': 1, 'PsBh': .5})]:
                rep(f'v3-04 C{cw} ' + extra_name, {**V304, 'C': cw, **extra})
        # small stacked grid that keeps all v3-04 members (each >= 0.25)
        print('=== grid (keeps Sd, C, Dr, xSL, consP8)', flush=True)
        g = []
        for sd, c, dr, cp8, crp, n2, pb, xs in itertools.product([.5, 1], [.25, .5, 1], [1, 1.5], [.5, 1],
                                                                 [1, 2, 3], [0, .5, 1], [0, .5], [1, 1.5]):
            w = {'Sd': sd, 'C': c, 'Dr': dr, 'xSL': xs, 'consP8': cp8, 'consRplus': crp, 'N2': n2, 'PsBh': pb}
            w = {a: b for a, b in w.items() if b}
            r = lco(w, views)
            g.append((r['m'], min(r['sets']), r['sets'], r['exc'], w))
            if time.time() - t0 > 520:
                print('grid time budget', len(g), flush=True); break
        g.sort(key=lambda x: -x[0])
        for m, mn, sets, exc, w in g[:10]:
            print(f'  LCO={m:.4f} min={mn:.3f} sets={np.round(sets, 3).tolist()} exc={np.round(exc, 2).tolist()} {w}')
        g.sort(key=lambda x: -x[1])
        print('  by worst set:')
        for m, mn, sets, exc, w in g[:5]:
            print(f'  LCO={m:.4f} min={mn:.3f} sets={np.round(sets, 3).tolist()} exc={np.round(exc, 2).tolist()} {w}')
        (O2 / f'recipes_{args.views.replace(",", "+")}.json').write_text(json.dumps(dict(rows=rows, grid=g[:100])))


if __name__ == '__main__':
    main()
