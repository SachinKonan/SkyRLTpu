import json, sys
for p in sys.argv[1:]:
    r = json.load(open(p + '/proposer_view.json'))
    if not r['ok']:
        print(p, 'FAIL', r.get('contract')); continue
    v = r['visible']
    print(f"{p.split('/')[-1]}: reward {r['reward']:+.5f} hidden_ok {r['hidden_ok']} folds {v['per_fold_gain']} cfg {r['config']} rt {r['runtime']['sec_per_1k_maps']}")
    print('  shifts', {k: x for k, x in sorted(v['per_shift_gain'].items(), key=lambda kv: -abs(kv[1]))})
    print('  standalone', v['standalone_visible_mean'], 'tail', r['tail']['exc_cand_pct'], 'vs', r['tail']['exc_base_pct'])
    print('  configs', [(c['misfit_weight'], c['vote'], c['reward']) for c in v['configs']])
