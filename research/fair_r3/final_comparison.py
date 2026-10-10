"""Normal-validation-selected noise models and simple spatial-specialist ensembles."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from compare import read_npz, load_run, average, normalized, inspect, paired_bootstrap
from common import write_json, sha


def main(a):
    root, out = Path(a.evidence), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    comparison = root / 'comparison-001'
    old = read_npz(comparison / '04-summary.npz')
    hybrid = read_npz(comparison / '05-summary-field.npz')
    selection = []; parts = []
    for seed in range(3):
        candidates = [root / 'checkpoint-001' / f'flow-noise4-s{seed}',
                      root / 'long-training-001' / f'flow-noise4-long-s{seed}']
        best = min(candidates, key=lambda p: json.loads((p / 'best.json').read_text())['val_conditional_nll'])
        selection.append(dict(seed=seed, selected=best.name, checkpoint_sha256=sha(best / 'best.msgpack'),
            candidates={p.name: json.loads((p / 'best.json').read_text()) for p in candidates}))
        parts.append(load_run(best))
    noise = average(parts); noise_normalized = normalized(noise); old_normalized = normalized(old)
    field = {k: 2 * v - old_normalized[k] for k, v in hybrid.items()}
    new_hybrid = {k: .5 * (noise_normalized[k] + v) for k, v in field.items()}
    regional = read_npz(root / 'spatial-residuals-001/regional-residual-scores.npz')
    region_normalized = normalized(regional)
    three_equal = {k: (2 * new_hybrid[k] + region_normalized[k]) / 3 for k in new_hybrid}
    hybrid_normalized = normalized(new_hybrid)
    region_max = {k: np.maximum(hybrid_normalized[k], region_normalized[k]) for k in new_hybrid}
    scores = {'04-summary': old, '05-summary-field': hybrid, 'noise-long-summary': noise,
              'noise-long-field': new_hybrid, 'regional-specialist': regional,
              'summary-field-regional-equal': three_equal, 'hybrid-regional-max': region_max}
    results = {k: inspect(v) for k, v in scores.items()}
    for name, values in scores.items():
        np.savez(out / (name + '.npz'), **values)
    rows = [dict(probe=k, **{name: r['probes'].get(k) for name, r in results.items()})
            for k in results['04-summary']['probes']]
    with (out / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    differences = {}
    for name in ('noise-long-field', 'summary-field-regional-equal', 'hybrid-regional-max'):
        differences[name] = {k: paired_bootstrap(hybrid, scores[name], k, a.bootstrap)
                            for k in hybrid if '/' in k and not k.endswith(('/normal', '/second'))}
    report = dict(kind='R3_FOLLOWUP_REUSED_DIAGNOSTICS_NOT_OFFICIAL_OOD', public_test_read=False,
                  noise_training_selection=selection, selection_basis='Minimum normal validation NLL per seed',
                  results=results, paired_differences_from_05=differences,
                  note='Ensemble choices are exploratory and normal-scaled; all synthetic families are reused development diagnostics.')
    write_json(out / 'report.json', report)
    for name, value in results.items():
        print(json.dumps(dict(name=name, normal_audit_1pct=value['normal']['exceedances'][1],
            worst_cosmology=value['normal']['max_cosmology_count_at_1pct'],
            scores={k: value['probes'].get(k) for k in (
                'development/spectral_low:0.2', 'development/quadratic_peak:0.2',
                'morphology_r2/phase_iaaft:1.0', 'r1_novel_now_reused/spatial_modulation:0.5')})), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--evidence', required=True)
    p.add_argument('--out', required=True); p.add_argument('--bootstrap', type=int, default=500)
    main(p.parse_args())
