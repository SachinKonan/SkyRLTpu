"""Exploratory replacement of candidate 05's summary part by the R3 noise ablation.

Reconstruct the frozen field component algebraically from recorded paired scores;
this changes no field weights, regressor or trace probe. No public data are read.
"""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'fair_r2'))
from common import low_fpr_metric, score_report, load_labels, splits, write_json
from compare import read_npz, normalized, paired_bootstrap


def groups(part, labels):
    ix = splits(); result = {}
    threshold = float(np.quantile(part['calibration'], .99))
    for col, name in enumerate(('Omega_m', 'S8', 'T_AGN', 'f0', 'delta_z')):
        edges = np.unique(np.quantile(labels[ix['train'], col], [0, .2, .4, .6, .8, 1]))
        bins = np.searchsorted(edges[1:-1], labels[ix['audit'], col], side='right')
        result[name] = [dict(lower=float(edges[i]), upper=float(edges[i + 1]),
                            count=int((bins == i).sum()),
                            flagged=int((part['audit'][bins == i] > threshold).sum()),
                            fraction=float((part['audit'][bins == i] > threshold).mean()))
                        for i in range(len(edges) - 1)]
    return result


def main(a):
    root, out = Path(a.comparison), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    old = read_npz(root / '04-summary.npz')
    hybrid = read_npz(root / '05-summary-field.npz')
    new = read_npz(root / 'R3-noise4.npz')
    old_scaled, new_scaled = normalized(old), normalized(new)
    field = {k: 2 * v - old_scaled[k] for k, v in hybrid.items()}
    revised = {k: .5 * (new_scaled[k] + v) for k, v in field.items()}
    np.savez(out / 'noise4-field-equal.npz', **revised)
    np.savez(out / 'frozen-field-normalized.npz', **field)
    labels = load_labels(a.raw)
    parts = {'04-summary': old, '05-summary-field': hybrid,
             'R3-noise4': new, 'R3-noise4-field': revised}
    report = dict(kind='EXPLORATORY_REUSED_DIAGNOSTICS_NOT_OFFICIAL_OOD',
        public_test_read=False, model_fitting=False,
        rule='Replace only candidate 05 summary score; field component and equal weight stay fixed; normal medians and 90th percentiles define scales.',
        normal={name: score_report({'score': p['calibration']}, {'score': p['audit']})['score']
                for name, p in parts.items()},
        known_parameter_false_positives={name: groups(p, labels) for name, p in parts.items()},
        differences_from_05={k: paired_bootstrap(hybrid, revised, k, a.bootstrap)
                             for k in hybrid if '/' in k and not k.endswith(('/normal', '/second'))},
        caveats=['These scores use previously examined synthetic families and maps.',
                 'Normal bins are descriptive; each has about 646 cases, with discrete cosmology ties.',
                 'This is a revised heuristic ensemble, not an exact likelihood ratio.'])
    rows = [dict(probe=k, **{name: low_fpr_metric(p['calibration'], p[k])
                             for name, p in parts.items()}) for k in hybrid if '/' in k]
    report['probes'] = rows
    write_json(out / 'report.json', report)
    with (out / 'comparison.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--comparison', required=True); p.add_argument('--out', required=True)
    p.add_argument('--raw', required=True); p.add_argument('--bootstrap', type=int, default=500)
    main(p.parse_args())
