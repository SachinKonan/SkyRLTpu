"""Pinned v1 score conventions. Only trusted graders construct these results."""
import math
import numpy as np

FLOOR = 1e-6


def positive(value):
    value = float(value)
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('reward must be finite and bounded')
    return max(FLOOR, value)


def qubit(baseline, candidate, weights):
    b, c, w = (np.asarray(x, dtype=np.float64) for x in (baseline, candidate, weights))
    if b.ndim != 1 or not len(b) or b.shape != c.shape or b.shape != w.shape:
        raise ValueError('case arrays must be nonempty and aligned')
    if not all(np.isfinite(x).all() for x in (b, c, w)) or (b < 0).any() or (c < 0).any() or (w <= 0).any():
        raise ValueError('invalid costs or case weights')
    B, C = float(w @ b), float(w @ c)
    if B <= 0:
        raise ValueError('baseline weighted added-CNOT count must be positive')
    return positive(B / (B + C)), dict(weighted_baseline_cnots=B, weighted_candidate_cnots=C,
        added_cnots=float(c.sum()), swaps=float(c.sum()/3), improvement=1-C/B)


def placement(baseline, candidate):
    b, c = (np.asarray(x, dtype=np.float64) for x in (baseline, candidate))
    if b.ndim != 2 or not len(b) or b.shape[1] != 3 or b.shape != c.shape:
        raise ValueError('expected aligned [cases, W/G/D] arrays')
    if not np.isfinite(b).all() or not np.isfinite(c).all() or (b <= 0).any() or (c < 0).any():
        raise ValueError('invalid placement costs or nonpositive baseline components')
    costs = (c / b) @ np.array([0.5, 0.4, 0.1])
    mean = float(costs.mean())
    return positive(1 / (1 + mean)), dict(placement_cost=mean, case_costs=costs.tolist(),
        wirelength=c[:,0].tolist(), congestion=c[:,1].tolist(), density=c[:,2].tolist())


def portfolio(net_returns, cash_returns, turnover):
    r, cash, turn = (np.asarray(x, dtype=np.float64) for x in (net_returns, cash_returns, turnover))
    if r.ndim != 1 or len(r) < 2 or r.shape != cash.shape or r.shape != turn.shape:
        raise ValueError('Sharpe requires at least two aligned daily observations')
    if not all(np.isfinite(x).all() for x in (r,cash,turn)) or (r <= -1).any() or (turn < 0).any():
        raise ValueError('nonfinite or impossible replay results')
    excess = r - cash
    std = float(excess.std(ddof=1))
    sharpe = float(math.sqrt(252) * excess.mean() / max(std, 1e-8))
    # Stable sigmoid, including extremely negative valid scientific scores.
    z = math.exp(-abs(sharpe))
    reward = 1 / (1 + z) if sharpe >= 0 else z / (1 + z)
    wealth = np.cumprod(np.r_[1., 1+r])
    return positive(reward), dict(sharpe=sharpe, cumulative_return=float(wealth[-1]-1),
        max_drawdown=float((1-wealth/np.maximum.accumulate(wealth)).max()),
        annualized_volatility=float(r.std(ddof=1)*math.sqrt(252)),
        excess_daily_std=std, low_volatility=std < 1e-8,
        mean_turnover=float(turn.mean()), days=len(r))


def valid(reward, metrics):
    score = positive(reward)
    return dict(reward=score, correctness=1., raw_score=score, msg='Valid',
                result_construction=None, stdout='', metrics=metrics)


def invalid(message, *, phase='candidate'):
    return dict(reward=0., correctness=0., raw_score=0., msg=str(message)[:4000],
                result_construction=None, stdout='', metrics=dict(failure_phase=phase))
