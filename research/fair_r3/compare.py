"""Compare R3 ablations on reused synthetic diagnostics; no public-test reads."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'fair_r2'))
from common import low_fpr_metric, score_report, write_json, sha


def read_npz(path):
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}


def baseline(root):
    r2 = root
    original = read_npz(r2 / 'comparison-003/flow-cosmology__mean_energy.npz')
    original.update({'morphology_r2/' + k: v for k, v in read_npz(
        r2 / 'checkpoint-004/morphology-evaluations/ensemble-cosmology.npz').items()})
    hybrid = read_npz(r2 / 'field-summary-ensembles-001/nll__hetero-absolute-equal.npz')
    hybrid.update(read_npz(r2 / 'blindspots-comparison-001/hybrid_scores.npz'))
    hybrid = {k.replace('reused/', 'r1_novel_now_reused/').replace(
        'morphology/', 'morphology_r2/'): v for k, v in hybrid.items()}
    return original, hybrid


def load_run(path):
    values = {k: read_npz(path / (k + '_scores.npz'))['minimum_base_energy']
              for k in ('calibration', 'audit')}
    for phase in ('development', 'r1_novel_now_reused', 'morphology_r2'):
        values.update({phase + '/' + k.removesuffix('__minimum_base_energy'): v
                       for k, v in read_npz(path / (phase + '_scores.npz')).items()
                       if k.endswith('__minimum_base_energy')})
    return values


def average(parts):
    keys = set(parts[0])
    if any(set(p) != keys for p in parts):
        raise ValueError('Unmatched score keys')
    return {k: np.stack([p[k] for p in parts]).mean(0) for k in parts[0]}


def normalized(part):
    mid, high = np.quantile(part['calibration'], [.5, .9])
    return {k: (v - mid) / max(high - mid, 1e-8) for k, v in part.items()}


def inspect(part):
    for k, v in part.items():
        if v.ndim != 1 or not np.isfinite(v).all():
            raise ValueError('Invalid scores: ' + k)
    report = dict(normal=score_report({'score': part['calibration']},
                                     {'score': part['audit']})['score'],
                  probes={k: low_fpr_metric(part['calibration'], v)
                          for k, v in part.items() if '/' in k})
    report['noise_repeatability'] = {
        phase: float(spearmanr(part[phase + '/normal'], part[phase + '/second']).statistic)
        for phase in ('development', 'r1_novel_now_reused', 'morphology_r2')
        if phase + '/second' in part}
    return report


def paired_bootstrap(control, alternative, key, n=500):
    # Shared indices within each normal/positive sample set preserve pairing.
    # IID conditional intervals do not include model selection or unknown OOD.
    rng = np.random.default_rng(202610093)
    differences = []
    for _ in range(n):
        ci = rng.integers(len(control['calibration']), size=len(control['calibration']))
        pi = rng.integers(len(control[key]), size=len(control[key]))
        differences.append(low_fpr_metric(alternative['calibration'][ci], alternative[key][pi]) -
                           low_fpr_metric(control['calibration'][ci], control[key][pi]))
    return dict(delta=low_fpr_metric(alternative['calibration'], alternative[key]) -
                low_fpr_metric(control['calibration'], control[key]),
                percentile95=np.quantile(differences, [.025, .975]).tolist(), replicates=n)


def main(a):
    root, out = Path(a.root), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    scores = dict(zip(('04-summary', '05-summary-field'), baseline(Path(a.r2))))
    sources, seed_reports = {}, {}
    for variant in ('morph', 'regional', 'joint', 'noise4'):
        parts = []
        for seed in range(3):
            path = root / f'flow-{variant}-s{seed}'
            if not (path / 'report.json').is_file():
                raise ValueError('Incomplete run: ' + str(path))
            values = load_run(path)
            seed_reports[path.name] = inspect(values)
            sources[path.name] = dict(checkpoint_sha256=sha(path / 'best.msgpack'),
                config=json.loads((path / 'config.json').read_text()),
                best=json.loads((path / 'best.json').read_text()),
                complete=json.loads((path / 'complete.json').read_text()))
            parts.append(values)
        scores['R3-' + variant] = average(parts)
    # Report two simple ensembles as exploratory results, not preselected winners.
    scores['R3-noise4+morph-equal'] = average([normalized(scores['R3-noise4']),
                                            normalized(scores['R3-morph'])])
    scores['R3-noise4+regional-equal'] = average([normalized(scores['R3-noise4']),
                                               normalized(scores['R3-regional'])])
    reports = {k: inspect(v) for k, v in scores.items()}
    keys = list(reports['04-summary']['probes'])
    rows = [dict(probe=k, **{name: r['probes'].get(k) for name, r in reports.items()}) for k in keys]
    with (out / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    for name, values in scores.items():
        np.savez(out / (name + '.npz'), **values)
    intervals = {}
    for name in ('R3-morph', 'R3-regional', 'R3-joint', 'R3-noise4'):
        intervals[name] = {k: paired_bootstrap(scores['04-summary'], scores[name], k, a.bootstrap)
                           for k in keys if not k.endswith(('/normal', '/second'))}
    write_json(out / 'report.json', dict(
        kind='R3_REUSED_SYNTHETIC_DIAGNOSTICS_NOT_OFFICIAL_OOD',
        public_test_read=False, sources=sources, results=reports, per_seed=seed_reports,
        paired_differences_from_04=intervals,
        limitations=['These diagnostics have been reused during model development.',
                     'Synthetic shifts do not identify the hidden physical alternatives.',
                     'Bootstrap conditions on fixed models and empirical IID sample sets; it excludes selection bias and unseen simulator uncertainty.',
                     'Normal audit counts are only 32 per cosmology.',
                     'Validation likelihoods across different feature spaces are not comparable.',
                     'Noise4 holds the original normalization fixed; views share clean realizations.',
                     'The two additional equal-weight ensembles are exploratory, added after registration.']))
    for name, report in reports.items():
        print(json.dumps(dict(name=name, normal_audit_1pct=report['normal']['exceedances'][1],
            worst_cosmology=report['normal']['max_cosmology_count_at_1pct'],
            selected={k: report['probes'].get(k) for k in (
                'development/spectral_low:0.2', 'development/spectral_high:0.2',
                'development/quadratic_peak:0.2', 'r1_novel_now_reused/spatial_modulation:0.5',
                'morphology_r2/phase_iaaft:1.0')})), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--r2', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--bootstrap', type=int, default=500)
    main(p.parse_args())
