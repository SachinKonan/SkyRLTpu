import numpy as np
from sklearn.metrics import roc_curve
from fairdiscover.metric import low_fpr_tpr
from fairdiscover.shifts import proxy_shift, FAMILIES


def test_exact_organizer_metric():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(1300), np.ones(900)]
    s = np.r_[rng.normal(size=1300), rng.normal(loc=1.5, size=900)]
    fpr, tpr, _ = roc_curve(y, s)
    target = np.interp(np.logspace(np.log10(.001), np.log10(.05), 100), fpr, tpr).mean()
    assert abs(low_fpr_tpr(y, s) - target) < 1e-12
    assert 0 <= low_fpr_tpr(y, s) <= 1


def test_ood_shifts_keep_observation_mask():
    rng = np.random.default_rng(7)
    x = rng.normal(0, .02, (16,12)).astype(np.float32)
    mask = np.ones((16,12), dtype=bool)
    mask[0] = False
    for family in FAMILIES:
        y = proxy_shift(x, mask, family, np.random.default_rng(8))
        assert y.shape == x.shape
        assert np.isfinite(y).all()
        assert np.all(y[~mask] == 0)
