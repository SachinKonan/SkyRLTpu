import numpy as np
import pandas as pd
import pytest
from tpu.science.portfolio_v2_accounting import day_transition,replay_v2


def test_old_holdings_take_overnight_move_before_new_orders():
    # Half stock at prior close doubles overnight. Sell it at open; no exposure
    # to its later crash. Initial $1 becomes $1.50, then pays the selling fee.
    ret,held,turn,_=day_transition(np.array([.5,.5]),[1.,0.],[2.],[0.],[.1],[True],.001)
    assert turn==pytest.approx(2/3)
    assert ret==pytest.approx(1.5*(1-.001*2/3)-1)
    np.testing.assert_allclose(held,[1,0])


def test_split_and_spinoff_preserve_wealth_and_cash_distributions():
    # A 10% distributed child is sold at open; remaining parent is worth 90%.
    ret,held,_,_=day_transition(np.array([0.,1.]),[.1,.9],[.9],[.1],[1.],[True],0.)
    assert ret==pytest.approx(0)
    np.testing.assert_allclose(held,[.1,.9])
    ret,_,_,_=day_transition(np.array([0.,1.]),[0.,1.],[1.],[0.],[1.],[True],0.)
    assert ret==0


def test_terminal_settlement_is_cash_and_new_order_is_unfilled():
    ret,held,turn,rejected=day_transition(np.array([0.,1.]),[0.,1.],[0.],[.954],[1.],[False],.001)
    assert ret==pytest.approx(-.046)
    np.testing.assert_allclose(held,[1,0]);assert turn==0 and rejected==1
    with pytest.raises(ValueError,match='unsettled'):
        day_transition(np.array([0.,1.]),[1.,0.],[1.],[0.],[1.],[False],0.)


def test_future_open_cannot_change_observed_holdings():
    frames=[pd.DataFrame({'day':[i]}) for i in range(3)]
    market=dict(parent_overnight=np.ones((3,1)),cash_distribution=np.zeros((3,1)),
        intraday=np.array([[1.2],[1.],[1.]]),tradable=np.ones((3,1),bool),
        right_units_per_dollar=np.zeros((3,1)),forced_sale_per_dollar=np.zeros((3,1)))
    def run(m):
        seen=[]
        def act(obs):seen.append(obs['current_weights'].copy());return [.5,.5]
        replay_v2(frames,m,act,0.)
        return seen
    baseline=run(market)
    altered={k:v.copy() for k,v in market.items()};altered['parent_overnight'][1]=10
    changed=run(altered)
    np.testing.assert_allclose(baseline[:2],changed[:2])
    np.testing.assert_allclose(baseline[1],[.5/1.1,.6/1.1])
