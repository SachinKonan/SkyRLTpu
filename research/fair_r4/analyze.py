"""Compare profiling and fixed ensembles using only recorded research scores."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'fair_r3'))
from compare import read_npz, normalized, inspect, average, paired_bootstrap
from check_noise_hybrid import groups
from common import load_labels, splits, write_json, score_report


def load_run(path, key):
    scores = {phase: read_npz(path/(phase+'.npz'))[key]
              for phase in ('val', 'calibration', 'audit')}
    for phase in ('development', 'r1_novel_now_reused', 'morphology_r2'):
        values = read_npz(path/(phase+'.npz'))
        scores.update({phase+'/'+k.removesuffix('__'+key): v for k, v in values.items()
                       if k.endswith('__'+key)})
    return scores


def main(a):
    root, r3, out = Path(a.root), Path(a.r3), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    runs = [root/f'audit-s{s}' for s in range(3)]
    assert all((r/'report.json').exists() for r in runs)
    grid = average([load_run(r, 'grid_energy') for r in runs])
    continuous = average([load_run(r, 'continuous_energy') for r in runs])
    true = {phase: np.stack([read_npz(r/(phase+'.npz'))['true_energy'] for r in runs]).mean(0)
            for phase in ('val', 'calibration', 'audit')}
    original = read_npz(r3/'final-comparison-001/noise-long-summary.npz')
    matching = {k: float(np.max(abs(v-grid[k]))) for k, v in original.items()}
    assert max(matching.values()) < 2e-4
    hybrid = read_npz(r3/'final-comparison-001/noise-long-field.npz')
    field = {k: 2*v-normalized(grid)[k] for k, v in hybrid.items()}
    continuous_hybrid = {k: .5*(normalized(continuous)[k]+v) for k, v in field.items()}
    regional = normalized(read_npz(r3/'regional-calibration-001/regional-centered.npz'))
    spatial = {k: np.maximum(normalized(continuous_hybrid)[k], regional[k]) for k in continuous_hybrid}
    original_spatial = read_npz(r3/'regional-calibration-001/hybrid+centered-max.npz')
    scores = {'grid-summary': grid, 'continuous-summary': continuous,
              '06-grid-hybrid': hybrid, 'continuous-hybrid': continuous_hybrid,
              '07-grid-spatial': original_spatial, 'continuous-spatial': spatial}
    reports = {k: inspect(v) for k, v in scores.items()}
    rows = [dict(probe=k, **{name: r['probes'].get(k) for name, r in reports.items()})
            for k in reports['grid-summary']['probes']]
    with (out/'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for name, part in scores.items(): np.savez(out/(name+'.npz'), **part)
    np.savez(out/'known-true-cosmology-NORMAL-ONLY.npz', **true)
    labels = load_labels(a.raw)
    bins = {name: groups(part, labels) for name, part in
            {'known-true-cosmology': true, **scores}.items()}
    differences = {}
    for first, second in [('grid-summary', 'continuous-summary'),
                          ('06-grid-hybrid', 'continuous-hybrid'),
                          ('07-grid-spatial', 'continuous-spatial')]:
        differences[second] = {k: paired_bootstrap(scores[first], scores[second], k, a.bootstrap)
                              for k in scores[second] if '/' in k and not k.endswith(('/normal', '/second'))}
    reduction = grid['audit']-continuous['audit']
    counts = {}
    for name, part in [('true', true), ('grid', grid), ('continuous', continuous)]:
        means = part['audit'].reshape(101, 32).mean(1)
        counts[name] = dict(
            overall_mean=float(part['audit'].mean()), overall_std=float(part['audit'].std()),
            mean_per_cosmology=means.tolist(),
            normal=score_report({'energy': part['calibration']}, {'energy': part['audit']})['energy'])
    report = dict(
        kind='R4_NORMAL_AND_REUSED_SYNTHETIC_NOT_OFFICIAL_OOD', public_test_read=False,
        results=reports, known_parameter_false_positives=bins, energy_diagnostics=counts,
        grid_reproduction_errors=matching,
        grid_continuous_spearman=float(spearmanr(grid['audit'], continuous['audit']).statistic),
        audit_energy_reduction_quantiles=np.quantile(reduction, [0, .5, .9, .99, 1]).tolist(),
        paired_differences=differences,
        limitations=['True cosmology is known only for normal diagnostics.',
                     'Synthetic families and normal audit maps have been reused.',
                     'IID paired bootstrap intervals condition on the selected models and probes; they exclude selection uncertainty.',
                     'Continuous interpolation enlarges the accepted physical model and can reduce power.'])
    write_json(out/'report.json', report)
    summary = {name: dict(audit_1pct=r['normal']['exceedances'][1],
                         low_omega_1pct=bins[name]['Omega_m'][0]['fraction'],
                         mild_low=r['probes']['development/spectral_low:0.2'],
                         mild_high=r['probes']['development/spectral_high:0.2'],
                         peak=r['probes']['development/quadratic_peak:0.2'],
                         morphology=r['probes']['morphology_r2/phase_iaaft:1.0'],
                         spatial=r['probes']['r1_novel_now_reused/spatial_modulation:0.5'])
               for name, r in reports.items()}
    print(json.dumps(dict(results=summary, normal_true_means={k: v['overall_mean'] for k, v in counts.items()},
                         rank_correlation=report['grid_continuous_spearman']), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'r3', 'out', 'raw'): p.add_argument('--'+name, required=True)
    p.add_argument('--bootstrap', type=int, default=500)
    main(p.parse_args())
