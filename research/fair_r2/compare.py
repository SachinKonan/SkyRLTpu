"""Compare fixed normal-calibrated scores on REUSED synthetic probes.

This is research selection evidence, not a fresh holdout or official OOD score.
No public-test file is read. Ensemble weights are equal, scales normal-only.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from common import low_fpr_metric, score_report, write_json, sha


def load_run(path, key):
    result = {}
    for phase in ('calibration', 'audit', 'development', 'r1_novel_now_reused'):
        with np.load(path / (phase + '_scores.npz')) as f:
            if phase in ('calibration', 'audit'):
                result[phase] = f[key]
            else:
                for k in f.files:
                    if k.endswith('__' + key):
                        result[phase + '/' + k.removesuffix('__' + key)] = f[k]
    return result


def reduce_scores(parts, fn):
    if any(set(p) != set(parts[0]) for p in parts):
        raise ValueError('Unmatched probe keys')
    return {k: fn(np.stack([p[k] for p in parts])) for k in parts[0]}


def calibrated_scores(part, two_sided=False):
    q10, mid, q90 = np.quantile(part['calibration'], [.1, .5, .9])
    upper = max(q90 - mid, 1e-8)
    lower = max(mid - q10, 1e-8)
    return {k: np.where(v >= mid, (v - mid) / upper,
                       (mid - v) / lower) if two_sided else (v - mid) / upper
            for k, v in part.items()}


def read_r1(root, directory):
    path = root / directory
    result = {k: np.load(path / 'density' / (k + '_scores.npy'))
              for k in ('calibration', 'audit')}
    for old, new in [('development', 'development'), ('novel', 'r1_novel_now_reused')]:
        with np.load(path / (old + '_diagnostic_scores.npz')) as f:
            result.update({new + '/' + k: f[k] for k in f.files})
    return result


def main(a):
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    scores, sources = {}, {}
    for prefix in ('flow', 'flow88', 'flow216'):
        for condition in ('none', 'cosmology', 'power'):
            dirs = sorted(root.glob(f'{prefix}-{condition}-*'))
            dirs = [d for d in dirs if (d / 'report.json').exists()]
            if not dirs:
                continue
            energy_key = 'minimum_base_energy' if condition == 'cosmology' else 'base_energy'
            nll_key = 'marginal_nll' if condition == 'cosmology' else 'nll'
            energies, nlls = [], []
            for d in dirs:
                energy = load_run(d, energy_key)
                scores[d.name + '/energy'] = energy
                energies.append(energy)
                nlls.append(load_run(d, nll_key))
                sources[d.name] = {'report_sha256': sha(d / 'report.json'),
                                   'config': json.loads((d / 'config.json').read_text())}
            name = prefix + '-' + condition
            mean_energy = reduce_scores(energies, lambda x: x.mean(0))
            mean_nll = reduce_scores(nlls, lambda x: x.mean(0))
            scores[name + '/mean_energy'] = mean_energy
            scores[name + '/two_sided_energy'] = calibrated_scores(mean_energy, True)
            scores[name + '/mean_nll'] = mean_nll
            if len(nlls) >= 2:
                scores[name + '/waic'] = reduce_scores(nlls, lambda x: x.mean(0) + x.var(0))
                scores[name + '/nll_variance'] = reduce_scores(nlls, lambda x: x.var(0))
    for name in ('gaussian72', 'gaussian216'):
        if not (root / name / 'report.json').exists():
            continue
        for key in ('joint_nll', 'minimum_mahalanobis', 'power_nll',
                    'conditional_shape_nll', 'posterior_conditional_energy'):
            scores[name + '/' + key] = load_run(root / name, key)
    if a.r1:
        for directory in ('physics-distance', 'physics'):
            scores['r1/' + directory] = read_r1(Path(a.r1), directory)
    for flowname in ('flow-cosmology', 'flow216-cosmology'):
        fk = flowname + '/mean_energy'
        if fk not in scores:
            continue
        for other in ('gaussian72/posterior_conditional_energy',
                      'gaussian216/posterior_conditional_energy',
                      'flow-none/mean_energy', 'flow-power/mean_energy'):
            if other not in scores:
                continue
            parts = [calibrated_scores(scores[k]) for k in (fk, other)]
            scores[flowname + '+equal+' + other] = reduce_scores(parts, lambda x: x.mean(0))
            scores[flowname + '+max+' + other] = reduce_scores(parts, lambda x: x.max(0))
    reports, rows = {}, []
    for name, result in scores.items():
        cal = result['calibration']
        norm = score_report({'score': cal}, {'score': result['audit']})['score']
        probe = {k: low_fpr_metric(cal, v) for k, v in result.items()
                 if '/' in k and not k.endswith('/second')}
        correlations = {phase: float(spearmanr(result[phase + '/normal'],
                                              result[phase + '/second']).statistic)
                        for phase in ('development', 'r1_novel_now_reused')}
        reports[name] = dict(normal=norm, probes=probe, noise_repeatability=correlations)
        rows.append(dict(name=name, normal_audit_1pct=norm['exceedances'][1],
                         worst_cosmology_1pct=norm['max_cosmology_count_at_1pct'],
                         noise_spearman=correlations['development'], **probe))
        np.savez(out / (name.replace('/', '__').replace('+', '_') + '.npz'), **result)
    fields = list(rows[0])
    with (out / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    write_json(out / 'report.json', dict(kind='REUSED_SYNTHETIC_PROBES_NOT_OFFICIAL_OOD',
        ensemble_rule='equal arithmetic means or max after normal-calibration median/q90 scaling; WAIC uses population variance over initialization seeds',
        limitations=['Repeated evaluation is model development, not an independent final test.',
                     'Cosmology groups have only 32 audit maps each.',
                     'WAIC-style initialization ensemble is not posterior sampling.',
                     'Synthetic probe rankings need not match the competition.'],
        sources=sources, results=reports))
    for row in rows:
        print(json.dumps({k: row[k] for k in ('name', 'normal_audit_1pct', 'worst_cosmology_1pct',
              'noise_spearman', 'development/spectral_low:0.2', 'development/spectral_high:0.2',
              'development/quadratic_peak:0.2', 'r1_novel_now_reused/anisotropic:0.2')}))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--r1')
    main(p.parse_args())
