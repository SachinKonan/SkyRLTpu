"""Explain sparse-score tradeoffs with matched dense and rotation controls."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from analyze_sparse import load_latents
from compare import read_npz, average, normalized, inspect, paired_bootstrap
from common import write_json


def main(a):
    root, comparison, out = map(Path, (a.root, a.comparison, a.out))
    out.mkdir(parents=True, exist_ok=False)
    names = ('08', 'raw_energy', 'diagonal_energy', 'diagonal_max',
             '08+diagonal_bj-mean', '08+diagonal_top4-mean')
    methods = {name: read_npz(comparison/(name+'.npz')) for name in names}
    base = normalized(methods['08'])
    for name in ('raw_energy', 'diagonal_energy'):
        part = normalized(methods[name])
        methods['08+'+name+'-mean'] = {k: .5*(v+part[k]) for k, v in base.items()}
        methods['08+'+name+'-max'] = {k: np.maximum(v, part[k]) for k, v in base.items()}
    rotation_parts = [[] for _ in range(4)]
    norm_errors = []
    for seed in range(3):
        latents = load_latents(root/f'latents-s{seed}')
        moments = read_npz(comparison/f'moments-s{seed}.npz')
        z = {k: (v.astype(np.float64)-moments['mean'])/moments['std'] for k, v in latents.items()}
        for rotation in range(4):
            rng = np.random.default_rng(40400+rotation)
            q, r = np.linalg.qr(rng.normal(size=(72,72)))
            q *= np.sign(np.diag(r))[None, :]
            assert np.allclose(q.T@q, np.eye(72), atol=1e-12)
            rotated = {k: v@q for k, v in z.items()}
            errors = {k: float(np.max(abs((v*v).sum(1)-(z[k]**2).sum(1))))
                      for k, v in rotated.items()}
            assert all(np.allclose((v*v).sum(1), (z[k]**2).sum(1), rtol=1e-12, atol=1e-10)
                       for k, v in rotated.items())
            norm_errors.append(dict(seed=seed, rotation=rotation, max_absolute_error=max(errors.values())))
            rotation_parts[rotation].append({k: (v*v).max(1) for k, v in rotated.items()})
    rotation_correlations = {}
    for rotation, parts in enumerate(rotation_parts):
        name = f'rotated_max_{rotation}'; methods[name] = average(parts)
        rotation_correlations[name] = float(spearmanr(methods[name]['audit'],
                                                       methods['diagonal_max']['audit']).statistic)
    reports = {name: inspect(part) for name, part in methods.items()}
    control = methods['08+raw_energy-mean']; alternative = methods['08+diagonal_bj-mean']
    probe_keys = [k for k in control if '/' in k and not k.endswith(('/normal', '/second'))]
    differences = {k: paired_bootstrap(control, alternative, k, a.bootstrap) for k in probe_keys}
    rows = [dict(probe=k, **{name: r['probes'][k] for name, r in reports.items()}) for k in probe_keys]
    with (out/'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    for name, values in methods.items():
        np.savez(out/(name+'.npz'), **values)
    report = dict(kind='POST_RESULT_MATCHED_CONTROLS_NOT_BLIND_VALIDATION', public_test_read=False,
                  results=reports, bj_minus_matched_dense_mean=differences,
                  rotation_energy_checks=norm_errors,
                  audit_rank_correlations_original_vs_rotated_max=rotation_correlations,
                  rationale='Mixing in any summary component reweights the original summary/field ensemble. Rotation diagnoses dependence on arbitrary learned latent axes.',
                  limitations=['All maps and probe families are reused.',
                               'No rotation is selected for deployment.',
                               'No claim of chi-squared null or independent Gaussian latent coordinates.',
                               'Paired IID bootstrap conditions on the selected fixed models and datasets.'])
    write_json(out/'report.json', report)
    print(json.dumps(dict(rotation_correlations=rotation_correlations,
                          IAAFT={name: r['probes']['morphology_r2/phase_iaaft:1.0'] for name, r in reports.items()},
                          IAAFT_bj_minus_dense=differences['morphology_r2/phase_iaaft:1.0']), indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'comparison', 'out'): p.add_argument('--'+name, required=True)
    p.add_argument('--bootstrap', type=int, default=500)
    main(p.parse_args())
