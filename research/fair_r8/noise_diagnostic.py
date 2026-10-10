"""Normal-only Gaussian references and a privileged clean-image error bound."""
import argparse
import json
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(1, str(Path(__file__).resolve().parents[1] / 'fair_r2'))
from common import splits, sha, write_json

NOISE = .4 / np.sqrt(240.)


def coefficients(t, variance):
    total = (1-t)**2 * variance + t*t
    gain = (t - (1-t)*variance) / total
    residual_variance = variance / total
    return gain, residual_variance


def analytic_check():
    # Data -> noise orientation, including a nonzero data mean. The reference
    # is derived directly from covariance conditioning, independently of the
    # simplified residual-variance identity used above.
    rng = np.random.default_rng(81733)
    records = []
    for variance in (.08, .6, 1., 3.):
        data = .7 + np.sqrt(variance) * rng.normal(size=200000)
        epsilon = rng.normal(size=len(data))
        for t in (.05, .3, .5, .85, .99):
            total = (1-t)**2 * variance + t*t
            covariance = t - (1-t)*variance
            gain, conditional_variance = coefficients(t, variance)
            direct = 1 + variance - covariance**2 / total
            assert np.allclose(direct, conditional_variance, rtol=1e-12, atol=1e-12)
            xt = (1-t)*data + t*epsilon
            prediction = -.7 + gain * (xt-(1-t)*.7)
            error = epsilon-data-prediction
            relative = float(error.var()/conditional_variance - 1)
            corr = float(np.corrcoef(error, xt)[0, 1])
            assert abs(relative) < .015 and abs(corr) < .015
            records.append(dict(variance=variance, t=t, relative_variance_error=relative,
                                residual_input_correlation=corr))
    return records


def main(a):
    cache, audit, out = map(Path, (a.cache, a.audit, a.out))
    out.mkdir(parents=True, exist_ok=False)
    checks = analytic_check()
    start = time.monotonic()
    with np.load(cache/'normalization.npz') as f:
        norm = {k: f[k] for k in f.files}
    with np.load(audit/'paired-errors.npz') as f:
        saved = {k: f[k] for k in f.files}
    support = norm['support'].astype(bool)
    counts = norm['count']; groups = np.unique(counts[support])
    mean, scale = float(norm['mean']), float(norm['scale'])
    clean = np.load(cache/'clean.npy', mmap_mode='r')
    fit_ids = splits()['train']
    stats = {int(k): np.zeros(3) for k in groups}
    for lo in range(0, len(fit_ids), 32):
        batch = np.asarray(clean[fit_ids[lo:lo+32]], np.float64)
        for k in groups:
            x = (batch[:, (counts == k) & support]-mean)/scale
            stats[int(k)] += [x.size, x.sum(), np.square(x).sum()]
    coverage_mean = np.zeros(support.shape)
    coverage_variance = np.ones(support.shape)
    fit = {}
    noise_variance = NOISE**2 / np.maximum(counts, 1) / scale**2
    for k in groups:
        n, sm, ss = stats[int(k)]; m = sm/n; v = ss/n-m*m
        select = (counts == k) & support
        coverage_mean[select] = m
        coverage_variance[select] = v + noise_variance[select]
        fit[str(k)] = dict(pixels=int(select.sum()), training_values=int(n),
                           clean_mean_normalized=float(m), clean_variance_normalized=float(v),
                           observation_noise_variance_normalized=float(noise_variance[select][0]),
                           noise_fraction=float(noise_variance[select][0]/coverage_variance[select][0]))

    ids = saved['ids']; used = np.random.default_rng(9183).permutation(splits()['val'])[:128]
    expected = np.random.default_rng(884101).permutation(np.setdiff1d(splits()['val'], used))[:1024]
    assert np.array_equal(ids, expected)
    rng = np.random.default_rng(884102)
    c = np.asarray(clean[ids], np.float32)
    observed = ((c + rng.normal(size=c.shape).astype(np.float32) *
                 (NOISE/np.sqrt(np.maximum(counts, 1)))-mean)/scale)*support
    epsilon = rng.normal(size=observed.shape).astype(np.float32)*support
    uniform_t = rng.uniform(.001, .999, size=len(ids)).astype(np.float32)
    assert np.array_equal(uniform_t, saved['uniform_t'])
    c = ((c-mean)/scale)[:, support].astype(np.float64)
    x = observed.astype(np.float32)[:, support].astype(np.float64)
    e = epsilon[:, support].astype(np.float64)
    del observed, epsilon
    target = e-x
    mu = coverage_mean[support][None, :]
    var = coverage_variance[support][None, :]
    nv = noise_variance[support][None, :]
    times = [uniform_t] + [np.full(len(ids), t) for t in saved['grid_t']]
    references = {k: np.empty((len(ids), len(times))) for k in
                  ['unit_gaussian', 'coverage_gaussian', 'clean_oracle', 'oracle_expected']}
    for column, t in enumerate(times):
        for lo in range(0, len(ids), 16):
            sl = slice(lo, lo+16); tt = t[sl, None]
            xt = (1-tt)*x[sl]+tt*e[sl]
            gain, _ = coefficients(tt, 1.)
            references['unit_gaussian'][sl, column] = np.square(target[sl]-gain*xt).mean(1)
            gain, _ = coefficients(tt, var)
            pred = -mu+gain*(xt-(1-tt)*mu)
            references['coverage_gaussian'][sl, column] = np.square(target[sl]-pred).mean(1)
            gain, expected_variance = coefficients(tt, nv)
            pred = -c[sl]+gain*(xt-(1-tt)*c[sl])
            references['clean_oracle'][sl, column] = np.square(target[sl]-pred).mean(1)
            references['oracle_expected'][sl, column] = expected_variance.mean(1)
    errors = {**{k: v for k, v in saved.items() if k.startswith(('old-', 'w16-', 'w32-'))},
              **references}
    baseline = references['coverage_gaussian'].mean(0)
    oracle = references['clean_oracle'].mean(0)
    results = {name: dict(uniform_t_mse=float(v[:, 0].mean()),
                         grid_mse=v[:, 1:].mean(0).tolist(),
                         fraction_reference_oracle_gap_explained=(
                             (baseline-v.mean(0))/np.maximum(baseline-oracle, 1e-9)).tolist())
               for name, v in errors.items()}
    np.savez(out/'paired-errors.npz', ids=ids, uniform_t=uniform_t, grid_t=saved['grid_t'], **errors)
    report = dict(kind='NORMAL_ONLY_PRIVILEGED_ORACLE_DIAGNOSTIC_NOT_DEPLOYABLE_SCORE',
                  public_test_read=False, synthetic_anomalies_read=False,
                  source_sha256=sha(__file__), audit_sha256=sha(audit/'paired-errors.npz'),
                  normalization_sha256=sha(cache/'normalization.npz'),
                  numerical_checks=checks, training_variance_by_coverage=fit,
                  results=results, seconds=time.monotonic()-start,
                  limitations=['The clean-image oracle uses unavailable inference-time information.',
                               'Gaussian references model independent pixels, not the correlated cosmological field.',
                               'This explains normal squared error, not unknown-anomaly sensitivity.',
                               'Per-time estimates and oracle bounds have finite-sample Monte Carlo error.'])
    write_json(out/'report.json', report)
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    for name in ('cache', 'audit', 'out'):p.add_argument('--'+name, required=True)
    main(p.parse_args())
