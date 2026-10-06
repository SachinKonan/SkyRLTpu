"""Trusted historical accounting; candidate receives one causal observation at a time."""
import numpy as np
from .contracts import weights
from .rewards import portfolio


def transition(current, target, asset_returns, cash_return, fee_rate):
    target = weights(target,len(asset_returns))
    all_returns = np.r_[cash_return,np.asarray(asset_returns,dtype=np.float64)]
    if not np.isfinite(all_returns).all() or (all_returns <= -1).any():
        raise ValueError('invalid market returns')
    turnover = float(np.abs(target[1:]-current[1:]).sum())
    cost = fee_rate*turnover
    if not 0 <= cost < 1:
        raise ValueError('invalid fee')
    gross = float(target @ all_returns)
    net = (1-cost)*(1+gross)-1
    drifted = target*(1+all_returns)/(1+gross)
    return net,drifted,turnover


def replay(data, act, fee_rate=0.001):
    features, returns, cash = (data[k] for k in ('features','returns','cash_returns'))
    context = data['context_features']
    if context.shape != (19,*features.shape[1:]) or returns.shape != features.shape[:2] or cash.shape != (len(features),):
        raise ValueError('invalid replay schema')
    history = np.concatenate([context,features])
    current = np.zeros(returns.shape[1]+1);current[0]=1
    net,turnover=[],[]
    for t in range(len(features)):
        # Copy ensures an in-process trusted baseline cannot alter future inputs.
        obs=dict(features=history[t:t+20].copy(),current_weights=current.copy())
        target = act(obs)
        value,current,turn=transition(current,target,returns[t],cash[t],fee_rate)
        net.append(value);turnover.append(turn)
    reward,metrics=portfolio(net,cash,turnover)
    return reward,metrics,dict(net_returns=net,turnover=turnover)
