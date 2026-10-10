"""Normal-only spatial residual control, using training-cosmology grouped CV."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'fair_r2'))
from common import load_features, splits, score_report, low_fpr_metric, write_json, sha


def fit_transform(x):
    offset = np.median(x, axis=0)
    scale = np.maximum(np.subtract(*np.quantile(x, [.75, .25], axis=0)) / 1.349, 1e-6)
    z = np.arcsinh((x - offset) / scale)
    return dict(offset=offset, scale=scale, mean=z.mean(0), std=np.maximum(z.std(0), 1e-6))


def transform(x, m):
    return (np.arcsinh((x - m['offset']) / m['scale']) - m['mean']) / m['std']


def ridge_fit(x, y, alpha):
    xm, ym = fit_transform(x), fit_transform(y)
    zx, zy = transform(x, xm), transform(y, ym)
    w = np.linalg.solve(zx.T @ zx + len(x) * alpha * np.eye(zx.shape[1]), zx.T @ zy)
    return dict(x=xm, y=ym, w=w)


def residual(x, y, m):
    return transform(y, m['y']) - transform(x, m['x']) @ m['w']


def predict(x, m):
    e = residual(x[:, :72], x[:, m['target_indices']], m) - m['center']
    return np.einsum('bi,ij,bj->b', e, m['precision'], e)


def save(path, m):
    np.savez(path, **{k: v for k, v in m.items() if k not in ('x', 'y')},
             **{f'{prefix}_{k}': v for prefix in ('x', 'y') for k, v in m[prefix].items()})


def load(path):
    with np.load(path) as f:
        m = {k: f[k] for k in f.files if not k.startswith(('x_', 'y_'))}
        for prefix in ('x', 'y'):
            m[prefix] = {k[2:]: f[k] for k in f.files if k.startswith(prefix + '_')}
    return m


def main(a):
    root, out = Path(a.features), Path(a.out)
    out.mkdir(parents=True, exist_ok=False)
    f = load_features(root); x = f['train'][:, :72]
    rng = np.random.default_rng(330)
    order = rng.permutation(101); assignment = np.empty(101, int)
    assignment[order] = np.arange(101) % 5
    folds = assignment[splits()['train'] // 256]
    probes = {}
    for path in (root / 'probes').glob('*.npz'):
        with np.load(path) as data:
            probes.update({path.stem + '/' + k: data[k].astype(np.float64)
                           for k in data.files if k != 'ids'})
    report = dict(kind='NORMAL_FIT_SPATIAL_RESIDUALS_REUSED_SYNTHETIC_DIAGNOSTICS',
                  public_test_read=False, results={}, models={},
                  limitations=['CV and covariance fitting use only normal training/validation maps.',
                               'A linear conditional mean and global residual covariance are approximations.',
                               'Power/shape conditioning can remove genuine OOD sensitivity.',
                               'This does not establish competition performance.'])
    for name, indices in (('morph', np.arange(72, 117)), ('regional', np.arange(117, 165))):
        y = f['train'][:, indices]; candidates = []
        for alpha in (.001, .01, .1, 1.):
            errors = []
            for fold in range(5):
                tr = folds != fold; va = ~tr
                m = ridge_fit(x[tr], y[tr], alpha)
                errors.append(float(np.mean(residual(x[va], y[va], m) ** 2)))
            candidates.append(dict(alpha=alpha, normal_cv_mse=float(np.mean(errors))))
        selected = min(candidates, key=lambda q: q['normal_cv_mse'])
        m = ridge_fit(x, y, selected['alpha'])
        r = residual(f['val'][:, :72], f['val'][:, indices], m)
        center = r.mean(0); cov = np.cov(r, rowvar=False)
        cov = .9 * cov + .1 * np.diag(np.diag(cov)) + 1e-8 * np.eye(len(indices))
        m.update(target_indices=indices, center=center, precision=np.linalg.inv(cov))
        path = out / (name + '-residual-model.npz'); save(path, m)
        fn = lambda v: predict(v, m)
        s = {k: fn(v) for k, v in {**{k: f[k] for k in ('calibration', 'audit')}, **probes}.items()}
        for v in s.values():
            assert np.isfinite(v).all() and (v >= -1e-8).all()
        check = f['val'][:17]
        assert np.array_equal(fn(check), predict(check, load(path)))
        assert np.allclose(fn(check), fn(check[::-1])[::-1], rtol=1e-12, atol=1e-12)
        assert np.allclose(fn(check)[:1], fn(check[:1]), rtol=1e-12, atol=1e-12)
        np.savez(out / (name + '-residual-scores.npz'), **s)
        report['models'][name] = dict(selected=selected, candidates=candidates,
                                      sha256=sha(path), training_count=len(x),
                                      covariance_normal_validation_count=len(r), shrinkage=.1,
                                      pointwise_and_serialization_checks=True)
        report['results'][name] = dict(
            normal=score_report({'score': s['calibration']}, {'score': s['audit']})['score'],
            probes={k: low_fpr_metric(s['calibration'], v) for k, v in s.items() if '/' in k})
        print(json.dumps(dict(model=name, selected=selected, results=report['results'][name])), flush=True)
    write_json(out / 'report.json', report)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--features', required=True); p.add_argument('--out', required=True)
    main(p.parse_args())
