"""Normal-only conditional score correction with cosmology-grouped validation."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from analyze import load_run
from compare import average, read_npz, normalized, inspect, paired_bootstrap
from check_noise_hybrid import groups
from common import load_labels, splits, write_json


def design(x, powers):
    return np.prod(np.asarray(x, np.float64)[:, None, :] ** powers[None, :, :], axis=2)


def fit(x, y, degree, alpha):
    powers = np.asarray([(i, n-i) for n in range(1, degree+1) for i in range(n+1)], int).reshape(-1, 2)
    z = design(x, powers); center = z.mean(0); scale = np.maximum(z.std(0), 1e-8)
    z = (z-center)/scale; bias = y.mean()
    weights = np.linalg.solve(z.T@z+len(z)*alpha*np.eye(z.shape[1]), z.T@(y-bias))
    return dict(powers=powers, center=center, scale=scale, weights=weights, bias=bias,
                degree=degree, alpha=alpha)


def predict(x, model):
    return (design(x, model['powers'])-model['center'])/model['scale']@model['weights']+model['bias']


def fit_variance(x, residual, degree, alpha, sd_floor):
    # Cross-fitted location residuals supply targets. A training-only factor
    # converts geometric squared-residual predictions to the arithmetic scale.
    m = fit(x, np.log(residual**2+(.1*sd_floor)**2), degree, alpha)
    raw = np.exp(np.clip(predict(x, m), -20, 20))
    m['variance_factor'] = float(np.mean(residual**2/raw))
    m['sd_floor'] = sd_floor
    return m


def predict_sd(x, model):
    variance = model['variance_factor']*np.exp(np.clip(predict(x, model), -20, 20))
    return np.sqrt(np.maximum(variance, model['sd_floor']**2))


def main(a):
    root, comparison, r3, out = map(Path, (a.root, a.comparison, a.r3, a.out))
    out.mkdir(parents=True, exist_ok=False)
    runs = [root/f'audit-s{s}' for s in range(3)]
    energy = average([load_run(r, 'continuous_energy') for r in runs])
    inputs = average([load_run(r, 'continuous_theta') for r in runs])
    x = inputs['val']; y = np.log(energy['val'].astype(np.float64))
    grouping = splits()['val']//256
    order = np.random.default_rng(4242).permutation(101)
    assignments = np.empty(101, int); assignments[order] = np.arange(101)%5
    folds = assignments[grouping]
    configs = [(d, alpha) for d in range(4) for alpha in
               ((1.,) if d == 0 else (.0001, .001, .01, .1, 1., 10.))]
    candidates = []; best = np.inf
    for degree, alpha in configs:
        oof = np.empty(len(y))
        for fold in range(5):
            tr = folds != fold
            oof[~tr] = predict(x[~tr], fit(x[tr], y[tr], degree, alpha))
        mse = float(np.mean((oof-y)**2))
        candidates.append(dict(degree=degree, alpha=alpha, normal_group_cv_mse=mse))
        if mse < best:
            best = mse; choice = (degree, alpha); location_oof = oof.copy()
    location = fit(x, y, *choice)
    residual = y-location_oof
    global_sd = float(np.sqrt(np.mean(residual**2))); sd_floor = .5*global_sd
    best = np.inf; variance_candidates = []
    for degree, alpha in configs:
        variance_oof = np.empty(len(y))
        for fold in range(5):
            tr = folds != fold
            model = fit_variance(x[tr], residual[tr], degree, alpha, sd_floor)
            variance_oof[~tr] = predict_sd(x[~tr], model)**2
        nll = float(.5*np.mean(np.log(variance_oof)+residual**2/variance_oof))
        variance_candidates.append(dict(degree=degree, alpha=alpha, normal_group_cv_gaussian_nll=nll))
        if nll < best: best = nll; variance_choice = (degree, alpha)
    variance = fit_variance(x, residual, *variance_choice, sd_floor)
    np.savez(out/'normal-score-regressors.npz',
             **{'location_'+k: v for k, v in location.items()},
             **{'variance_'+k: v for k, v in variance.items()}, global_sd=global_sd)
    np.savez(out/'normal-crossfit.npz', ids=splits()['val'], folds=folds,
             y=y, location_oof=location_oof, inferred_theta=x)

    centered = {k: np.log(v.astype(np.float64))-predict(inputs[k], location) for k, v in energy.items()}
    scaled = {k: v/predict_sd(inputs[k], variance) for k, v in centered.items()}
    two_sided = {k: abs(v) for k, v in scaled.items()}
    for values in (centered, scaled, two_sided):
        assert all(np.isfinite(v).all() for v in values.values())
    ref = scaled['val'][:27]
    def point_score(xx, ee): return (np.log(ee.astype(np.float64))-predict(xx, location))/predict_sd(xx, variance)
    cases = [point_score(x[:27][::-1], energy['val'][:27][::-1])[::-1],
             np.r_[point_score(x[:1], energy['val'][:1]), point_score(x[1:13], energy['val'][1:13]),
                   point_score(x[13:27], energy['val'][13:27])]]
    errors = [float(np.max(abs(v-ref))) for v in cases]
    assert max(errors)<1e-10

    hybrid = read_npz(comparison/'continuous-hybrid.npz')
    field = {k: 2*v-normalized(energy)[k] for k, v in hybrid.items()}
    region = normalized(read_npz(r3/'regional-calibration-001/regional-centered.npz'))
    scores = {'continuous-summary': energy, 'centered-summary': centered,
              'scaled-summary': scaled, 'two-sided-summary': two_sided,
              'continuous-hybrid': hybrid,
              'continuous-spatial': read_npz(comparison/'continuous-spatial.npz')}
    for name, values in [('centered', centered), ('scaled', scaled), ('two-sided', two_sided)]:
        sn = normalized(values)
        revised = {k: .5*(sn[k]+v) for k, v in field.items()}
        scores[name+'-hybrid'] = revised
        hn = normalized(revised)
        scores[name+'-spatial'] = {k: np.maximum(hn[k], region[k]) for k in hn}
    reports = {name: inspect(v) for name, v in scores.items()}
    labels = load_labels(a.raw)
    bins = {name: groups(v, labels) for name, v in scores.items()}
    rows = [dict(probe=k, **{name: r['probes'].get(k) for name, r in reports.items()})
            for k in reports['continuous-summary']['probes']]
    with (out/'comparison.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    for name, values in scores.items(): np.savez(out/(name+'.npz'), **values)
    differences = {}
    for name in ('centered', 'scaled', 'two-sided'):
        for component in ('summary', 'hybrid', 'spatial'):
            key = name+'-'+component; control = scores['continuous-'+component]
            differences[key] = {k: paired_bootstrap(control, scores[key], k, a.bootstrap)
                                for k in scores[key] if '/' in k and not k.endswith(('/normal', '/second'))}
    report = dict(kind='NORMAL_ONLY_SCORE_CORRECTION_REUSED_SYNTHETIC_DIAGNOSTICS',
                  public_test_read=False, location_candidates=candidates, selected_location=choice,
                  variance_candidates=variance_candidates, selected_variance=variance_choice,
                  location_cv_rmse_over_target_sd=global_sd/float(y.std()), sd_floor=sd_floor,
                  pointwise_errors=errors, results=reports, known_parameter_false_positives=bins,
                  paired_differences=differences,
                  limitations=['Previously examined normal audit and synthetic data.',
                               'Regressors fit only normal validation maps; those maps also selected flow checkpoints.',
                               'Location cross-fitting and conditional scale CV are not fully nested whole-pipeline CV.',
                               'Learned physical coordinates can absorb anomaly signals.',
                               'Score correction is heuristic and not an exact conditional p-value.'])
    write_json(out/'report.json', report)
    brief = {name: dict(audit_1pct=r['normal']['exceedances'][1],
                       low_omega_1pct=bins[name]['Omega_m'][0]['fraction'],
                       low=r['probes']['development/spectral_low:0.2'],
                       high=r['probes']['development/spectral_high:0.2'],
                       peaks=r['probes']['development/quadratic_peak:0.2'],
                       morphology=r['probes']['morphology_r2/phase_iaaft:1.0'],
                       spatial=r['probes']['r1_novel_now_reused/spatial_modulation:0.5'])
             for name, r in reports.items()}
    print(json.dumps(dict(location=choice, variance=variance_choice,
                         location_cv_rmse_over_target_sd=report['location_cv_rmse_over_target_sd'],
                         results=brief), indent=2), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('root', 'comparison', 'r3', 'raw', 'out'): p.add_argument('--'+name, required=True)
    p.add_argument('--bootstrap', type=int, default=500)
    main(p.parse_args())
