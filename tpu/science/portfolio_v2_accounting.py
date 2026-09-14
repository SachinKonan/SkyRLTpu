"""Reviewed corporate-action accounting, with causal close-time observations.

Market factors are trusted future outcomes. They must never be sent to a policy.
The policy decides at close; orders fill next open; daily PnL marks next close.
"""
import numpy as np
from .contracts import weights
from .rewards import portfolio


def day_transition(current, requested, parent_overnight, cash_distribution,
                   intraday, tradable, fee_rate):
    """Close-to-close wealth, including overnight old holdings and new orders."""
    n = len(current)-1
    requested = weights(requested, n)
    parent_overnight = np.asarray(parent_overnight, dtype=float)
    cash_distribution = np.asarray(cash_distribution, dtype=float)
    intraday = np.asarray(intraday, dtype=float)
    tradable = np.asarray(tradable, dtype=bool)
    if any(x.shape != (n,) for x in (parent_overnight,cash_distribution,intraday,tradable)):
        raise ValueError('market factor shape mismatch')
    if not all(np.isfinite(x).all() for x in (parent_overnight,cash_distribution,intraday)):
        raise ValueError('nonfinite market factor')
    if (parent_overnight < 0).any() or (cash_distribution < 0).any() or (intraday <= 0).any():
        raise ValueError('invalid market factor')
    opening_assets = current[1:] * parent_overnight
    opening_cash = current[0] + current[1:] @ cash_distribution
    opening_wealth = float(opening_cash + opening_assets.sum())
    if opening_wealth <= 0:
        raise ValueError('nonpositive portfolio wealth')
    if (opening_assets[~tradable] > 1e-12).any():
        raise ValueError('unsettled holding in unavailable security')
    opening_weights = np.r_[opening_cash, opening_assets] / opening_wealth
    # A broker cannot fill an order in a terminated security. The requested
    # capital remains cash; this execution outcome is not a future action mask.
    target = requested.copy()
    rejected_weight = float(target[1:][~tradable].sum())
    target[0] += rejected_weight
    target[1:][~tradable] = 0
    turnover = float(np.abs(target[1:]-opening_weights[1:]).sum())
    cost = fee_rate * turnover
    if not 0 <= cost < 1:
        raise ValueError('invalid transaction cost')
    closing = target * np.r_[1.,intraday]
    closing_factor = float(closing.sum())
    end_wealth = opening_wealth * (1-cost) * closing_factor
    return end_wealth-1, closing/closing_factor, turnover, rejected_weight


def replay_v2(observations, market, act, fee_rate=0.001):
    """observations[t] is the close strictly before market[t]'s opening."""
    n = market['parent_overnight'].shape[1]
    current = np.r_[1.,np.zeros(n)]
    net, turns, rejected, rights, forced = [], [], [], [], []
    if len(observations) != len(market['parent_overnight']):
        raise ValueError('observation and market length mismatch')
    for t, frame in enumerate(observations):
        # Crucially, current comes from the preceding CLOSE, never the next open.
        action = act(dict(market=frame.copy(deep=True), current_weights=current.copy()))
        rights.append(float(current[1:] @ market['right_units_per_dollar'][t]))
        forced.append(float(current[1:] @ market['forced_sale_per_dollar'][t]))
        value, current, turnover, rejected_weight = day_transition(current, action,
            market['parent_overnight'][t], market['cash_distribution'][t],
            market['intraday'][t], market['tradable'][t], fee_rate)
        net.append(value); turns.append(turnover); rejected.append(rejected_weight)
    reward, metrics = portfolio(net,np.zeros(len(net)),np.asarray(turns)+np.asarray(forced))
    metrics.update(rejected_order_weight_sum=float(sum(rejected)),
                   contingent_right_units_per_starting_dollar_note='trace entries are per current beginning-of-day dollar',
                   forced_sale_turnover=float(sum(forced)),evaluation_days=len(net))
    return reward,metrics,dict(net_returns=net,turnover=turns,rejected_order_weight=rejected,
                              contingent_right_units_per_day_dollar=rights,forced_sale_turnover=forced)
