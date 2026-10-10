"""Exploratory normal-only conditional centering/scaling of spatial residual scores."""
import argparse
import json
from pathlib import Path
import numpy as np
from spatial_residuals import load, predict, fit_transform, transform
from compare import read_npz, normalized, inspect
from common import load_features, splits, write_json, sha


def fit(x, y, alpha):
    m = fit_transform(x); z = transform(x, m); ym = y.mean()
    w = np.linalg.solve(z.T @ z + alpha * len(x) * np.eye(z.shape[1]), z.T @ (y - ym))
    return dict(**m, w=w, ym=ym, alpha=alpha)


def regress(x, m):
    return transform(x, m) @ m['w'] + m['ym']


def main(a):
    root, out = Path(a.features), Path(a.out); out.mkdir(parents=True, exist_ok=False)
    f = load_features(root); raw_scores = read_npz(a.scores); m = load(a.model)
    # Fixed feature map and spatial model were fit on train. Calibration/audit
    # never fit these regressors. Validation supplies new normal residuals.
    x = f['val'][:, :72]; y = np.log(np.maximum(predict(f['val'], m), 1e-12))
    groups = splits()['val'] // 256; order = np.random.default_rng(330).permutation(101)
    folds = np.empty(101, int); folds[order] = np.arange(101) % 5; fold = folds[groups]
    best = np.inf; candidates = []
    for alpha in (.001, .01, .1, 1.):
        oof = np.empty(len(y))
        for i in range(5):
            tr = fold != i; model = fit(x[tr], y[tr], alpha)
            oof[~tr] = regress(x[~tr], model)
        mse = float(np.mean((oof - y) ** 2)); candidates.append(dict(alpha=alpha, normal_cv_mse=mse))
        if mse < best:
            best = mse; chosen = alpha; held = oof.copy()
    mean = fit(x, y, chosen)
    variance_floor = (.1 * y.std()) ** 2
    variance = fit(x, np.log((y - held) ** 2 + variance_floor), .1)
    np.savez(out / 'normal-regressors.npz',
             **{f'mean_{k}': v for k, v in mean.items()},
             **{f'variance_{k}': v for k, v in variance.items()}, variance_floor=variance_floor)
    all_features = {k: f[k][:, :72] for k in ('calibration', 'audit')}
    for path in (root / 'probes').glob('*.npz'):
        with np.load(path) as data:
            all_features.update({path.stem + '/' + k: data[k][:, :72].astype(np.float64)
                                 for k in data.files if k != 'ids'})
    centered, scaled = {}, {}
    for k, raw in raw_scores.items():
        v = np.log(np.maximum(raw, 1e-12)) - regress(all_features[k], mean)
        sd = np.maximum(np.exp(np.clip(regress(all_features[k], variance), -20, 20) / 2),
                        np.sqrt(variance_floor))
        centered[k] = v; scaled[k] = v / sd
    summary = read_npz(Path(a.baseline) / 'noise-long-summary.npz')
    hybrid = read_npz(Path(a.baseline) / 'noise-long-field.npz')
    hn = normalized(hybrid)
    scores = {'regional-raw': raw_scores, 'regional-centered': centered, 'regional-scaled': scaled}
    for name, values in (('centered', centered), ('scaled', scaled)):
        rn = normalized(values)
        scores['hybrid+' + name + '-equal'] = {k: (2 * hybrid[k] + rn[k]) / 3 for k in hybrid}
        scores['hybrid+' + name + '-max'] = {k: np.maximum(hn[k], rn[k]) for k in hybrid}
    results = {k: inspect(v) for k, v in scores.items()}
    for name, values in scores.items(): np.savez(out / (name + '.npz'), **values)
    write_json(out / 'report.json', dict(kind='NORMAL_ONLY_CONDITIONAL_SPATIAL_CALIBRATION',
        public_test_read=False, candidates=candidates, selected_alpha=chosen,
        normal_cv_rmse_over_target_sd=float(np.sqrt(best) / y.std()), fit_count=len(y),
        source_model_sha256=sha(a.model), results=results,
        limitation='Exploratory heuristic conditional standardization; it need not remove all cosmology dependence. Synthetic evaluations are reused.'))
    for name, r in results.items():
        print(json.dumps(dict(name=name, normal_audit_1pct=r['normal']['exceedances'][1],
            worst_cosmology=r['normal']['max_cosmology_count_at_1pct'],
            probes={k: r['probes'].get(k) for k in ('development/spectral_low:0.2',
                'development/quadratic_peak:0.2', 'morphology_r2/phase_iaaft:1.0',
                'r1_novel_now_reused/spatial_modulation:0.5')})), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('features', 'scores', 'model', 'baseline', 'out'): p.add_argument('--' + name, required=True)
    main(p.parse_args())
