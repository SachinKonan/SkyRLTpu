import numpy as np


def fit(train,valid,rng,*,time_budget_s):
    return None


def act(model,obs):
    active=obs['market']['price_observed'].to_numpy(dtype=bool)
    weights=np.zeros(len(active)+1)
    if active.any():weights[1:]=active/active.sum()
    else:weights[0]=1.
    return weights
