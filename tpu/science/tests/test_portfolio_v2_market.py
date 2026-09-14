"""Preparation checks; run in requirements-data.lock environment."""
import numpy as np
import pandas as pd
import pytest
pytest.importorskip('duckdb')
from pandas.testing import assert_frame_equal
from tpu.science.portfolio_v2 import UNIVERSE,market_features,FEATURES
from tpu.science.prepare_portfolio_v2_runtime import accounting_panel,reconstruct_quotes


def fixture():
    dates=pd.bdate_range('2024-01-01','2024-01-05')
    schedule=pd.DataFrame(dict(open=dates.tz_localize('UTC')+pd.Timedelta(hours=14),
                              close=dates.tz_localize('UTC')+pd.Timedelta(hours=21)),index=dates)
    records=[]
    for asset in UNIVERSE:
        for day in dates:
            price=100.;volume=1000.
            if asset=='IBM' and day>=pd.Timestamp('2024-01-03'):price=90.
            if asset=='MRK' and day>=pd.Timestamp('2024-01-03'):price=50.
            if asset=='WBA' and day>=pd.Timestamp('2024-01-04'):price=np.nan;volume=0.
            records.append(dict(symbol='RTX' if asset=='UTX' else asset,session=day,
                                open=price,close=price,high=price+1,low=price-1,volume=volume))
    prices=pd.DataFrame(records)
    div=pd.DataFrame(dict(symbol=['IBM'],session=[dates[0]],amount=[0.]))
    panel=market_features(prices,div,schedule)
    quotes={symbol:g.set_index('session').assign(unit_scale=1.) for symbol,g in prices.groupby('symbol')}
    quotes['KD']=pd.DataFrame(dict(open=50.,volume=1000.),index=dates)
    actions=dict(splits=[dict(asset='MRK',date='2024-01-03',shares=2.)],
        spinoffs=[dict(asset='IBM',date='2024-01-03',children={'KD':.2})],
        terminal=[dict(asset='WBA',date='2024-01-04',cash_per_share=90.,right_units_per_share=1.,right_value=0.)])
    return panel,quotes,actions


def test_reviewed_split_spinoff_and_terminal_labels():
    panel,quotes,actions=fixture()
    frame,days,m,_=accounting_panel(panel,quotes,actions)
    i=days.get_loc('2024-01-03');ibm=list(UNIVERSE).index('IBM');mrk=list(UNIVERSE).index('MRK')
    assert m['parent_overnight'][i,ibm]==pytest.approx(.9)
    assert m['cash_distribution'][i,ibm]==pytest.approx(.0999)
    assert m['parent_overnight'][i,mrk]==pytest.approx(1.)
    row=frame[(frame.asset_id=='WBA')&(frame.session==pd.Timestamp('2024-01-02'))].iloc[0]
    assert row.target_return==pytest.approx(-.1) and row.label_released
    row=frame[(frame.asset_id=='IBM')&(frame.session==pd.Timestamp('2024-01-03'))].iloc[0]
    assert row.return_1d==pytest.approx(-.0001)


def test_unexplained_missing_quote_is_not_silently_scored():
    panel,quotes,actions=fixture()
    quotes['IBM'].loc['2024-01-04','open']=np.nan
    with pytest.raises(ValueError,match='unexplained missing price: IBM'):
        accounting_panel(panel,quotes,actions)


def test_corporate_accounting_prefix_matches_full_inputs():
    panel,quotes,actions=fixture();full,_,_,_=accounting_panel(panel,quotes,actions)
    cutoff=pd.Timestamp('2024-01-03')
    prefix,_,_,_=accounting_panel(panel[panel.session<=cutoff],
        {k:v.loc[:cutoff] for k,v in quotes.items()},actions)
    columns=['session','asset_id']+FEATURES[:9]
    assert_frame_equal(full.loc[full.session<=cutoff,columns].reset_index(drop=True),prefix[columns])


def test_future_split_backadjustment_reconstructs_identical_past_quotes():
    dates=pd.bdate_range('2024-01-01','2024-01-05')
    p=pd.DataFrame(dict(symbol='AAPL',session=dates,open=100.,close=102.,volume=1000.))
    empty=pd.DataFrame(columns=['symbol','report_date','split_factor'])
    original=reconstruct_quotes(p,empty)['AAPL']
    changed=p.copy();changed[['open','close']]/=4
    events=pd.DataFrame([dict(symbol='AAPL',report_date='2024-02-01',split_factor='4:1')])
    revised=reconstruct_quotes(changed,events)['AAPL']
    assert_frame_equal(original[['open','close']],revised[['open','close']])
