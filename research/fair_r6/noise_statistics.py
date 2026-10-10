"""Per-map normality and masked spatial whiteness diagnostics."""
import numpy as np
from scipy.special import log_ndtr


def anderson_darling(v):
    ordered = np.sort(np.asarray(v, np.float64), axis=1)
    dim = ordered.shape[1]
    weights = (2*np.arange(1, dim+1)-1)/dim
    return -dim-np.sum(weights*(log_ndtr(ordered)+log_ndtr(-ordered[:, ::-1])), axis=1)


def statistics(fields, support):
    support = np.asarray(support, bool)
    z = np.asarray(fields, np.float64)
    if z.ndim != 3 or z.shape[1:] != support.shape or not np.isfinite(z[:, support]).all():
        raise ValueError('Expected finite observed latent pixels with the declared support')
    v = z[:, support]; mean = v.mean(1); variance = v.var(1)
    standardized = (v-mean[:, None])/np.maximum(np.sqrt(variance[:, None]), 1e-12)
    centered = np.zeros_like(z); centered[:, support] = v-mean[:, None]
    power = abs(np.fft.fft2(centered, axes=(1,2)))**2
    # Full complex FFT gives the intended symmetric spectrum; omit only DC.
    power = power.reshape(len(z), -1)[:, 1:]
    cv = power.std(1)/np.maximum(power.mean(1), 1e-12)
    correlations = []; pair_counts = []
    for dy, dx in ((0,1), (1,0), (0,4), (4,0)):
        a = (slice(0, support.shape[0]-dy), slice(0, support.shape[1]-dx))
        b = (slice(dy, None), slice(dx, None))
        valid = support[a] & support[b]; n = int(valid.sum())
        if n == 0: raise ValueError('Support is too small for the registered lags')
        u = centered[(slice(None), *a)][:, valid]
        w = centered[(slice(None), *b)][:, valid]
        correlations.append((u*w).mean(1)/np.maximum(variance, 1e-12)); pair_counts.append(n)
    corr = np.column_stack(correlations)
    result = dict(energy=(v*v).sum(1), ad=anderson_darling(v),
                  ad_shape=anderson_darling(standardized), psd_cv=cv,
                  correlation_energy=(corr*corr*np.asarray(pair_counts)).sum(1),
                  mean=mean, variance=variance)
    result.update({f'correlation_{i}': corr[:,i] for i in range(corr.shape[1])})
    if not all(np.isfinite(x).all() for x in result.values()): raise ValueError('Invalid statistic')
    return result
