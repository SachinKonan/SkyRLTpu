import numpy as np


def fit(train, valid, rng, *, time_budget_s):
    return {'n': train['features'].shape[1]}


def act(model, obs):
    n = model['n']
    return np.r_[0., np.full(n, 1/n)]
