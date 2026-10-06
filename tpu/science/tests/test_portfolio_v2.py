"""Adversarial temporal tests, independent of any historical market outcomes."""
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from tpu.science.portfolio_v2 import (FEATURES,OBSERVATION_COLUMNS,conservative_available_at,
    normalize_news,market_features,join_news,join_treasury,observation,fit_frame,audit)


def data():
    dates=pd.bdate_range('2023-10-02','2024-01-12')
    schedule=pd.DataFrame({'open':dates.tz_localize('UTC')+pd.Timedelta(hours=14,minutes=30),
                           'close':dates.tz_localize('UTC')+pd.Timedelta(hours=21)},index=dates)
    close=np.linspace(100,120,len(dates))
    prices=pd.DataFrame(dict(symbol='AAPL',session=dates,open=close-.2,high=close+1,
                             low=close-1,close=close,volume=np.arange(len(dates))+1000))
    dividends=pd.DataFrame(dict(symbol=['AAPL'],session=[dates[30]],amount=[.2]))
    treasury=pd.DataFrame(dict(report_date=['2023-10-01','2023-12-29','2024-01-01'],
                               bc_3month=[5.,7.,9.]))
    return prices,dividends,treasury,schedule


def articles():
    return pd.DataFrame([
        dict(symbol='AAPL',title='old report',report_date='2023-12-26',source='one',link='a'),
        dict(symbol='AAPL',title='new report',report_date='2024-01-01T18:00:00Z',source='two',link='b'),
        dict(symbol='AAPL',title='same day report',report_date='2023-12-29',source='two',link='c')])


def construct(prices,dividends,treasury,schedule,news):
    return join_news(join_treasury(market_features(prices,dividends,schedule),treasury),news)


def test_conservative_date_only_and_bad_dates():
    assert conservative_available_at('2024-01-01')==pd.Timestamp('2024-01-03T00:00:00Z')
    assert conservative_available_at('2024-01-01T23:59:00-12:00')==pd.Timestamp('2024-01-03T00:00:00Z')
    assert pd.isna(conservative_available_at('not-a-date'))
    assert pd.isna(conservative_available_at('2024-02-30'))


def test_future_prices_dividends_and_news_do_not_change_past_inputs():
    prices,dividends,treasury,schedule=data()
    news,_=normalize_news(articles())
    original=construct(prices,dividends,treasury,schedule,news)
    cutoff=pd.Timestamp('2023-12-29')
    changed_prices=prices.copy()
    changed_prices.loc[changed_prices.session>cutoff,['open','high','low','close','volume']]*=8
    changed_dividends=pd.concat([dividends,pd.DataFrame(dict(symbol=['AAPL'],session=[pd.Timestamp('2024-01-04')],amount=[900.]))])
    changed_treasury=treasury.copy();changed_treasury.loc[changed_treasury.report_date>'2023-12-29','bc_3month']=100
    extra=pd.DataFrame([dict(symbol='AAPL',title='future headline revealing outcome',report_date='2024-01-05',source='z',link='z')])
    changed_news,_=normalize_news(pd.concat([articles(),extra]))
    modified=construct(changed_prices,changed_dividends,changed_treasury,schedule,changed_news)
    assert_frame_equal(original.loc[original.session<=cutoff,OBSERVATION_COLUMNS].reset_index(drop=True),
                       modified.loc[modified.session<=cutoff,OBSERVATION_COLUMNS].reset_index(drop=True))


def test_future_split_rescaling_cannot_change_price_features():
    prices,dividends,treasury,schedule=data()
    baseline=market_features(prices,dividends,schedule)
    revised=prices.copy();revised[['open','high','low','close']]/=4;revised['volume']*=4
    div=dividends.copy();div['amount']/=4
    actual=market_features(revised,div,schedule)
    assert_frame_equal(baseline[FEATURES[:9]],actual[FEATURES[:9]])


def test_prefix_build_matches_full_build():
    prices,dividends,treasury,schedule=data();news,_=normalize_news(articles())
    full=construct(prices,dividends,treasury,schedule,news)
    cutoff=pd.Timestamp('2023-12-29')
    prefix=construct(prices[prices.session<=cutoff],dividends[dividends.session<=cutoff],
        treasury[treasury.report_date<='2023-12-29'],schedule.loc[:cutoff],
        news[news.available_at<=pd.Timestamp('2023-12-29T21:15:00Z')])
    assert_frame_equal(full.loc[full.session<=cutoff,OBSERVATION_COLUMNS].reset_index(drop=True),
                       prefix[OBSERVATION_COLUMNS])


def test_boundaries_and_feature_allowlist():
    prices,dividends,treasury,schedule=data();news,_=normalize_news(articles())
    frame=construct(prices,dividends,treasury,schedule,news)
    assert audit(frame,news)['causal_news_edges_checked']>0
    train=fit_frame(frame,'train')
    assert train.loc[train.session.eq(pd.Timestamp('2023-12-29')),'target_return'].isna().all()
    obs=observation(frame,'2023-12-29')
    assert set(obs)==set(OBSERVATION_COLUMNS)
    assert all('target' not in c and 'label' not in c and 'execution' not in c for c in obs)
    assert 'same day report' not in obs.loc[obs.asset_id.eq('AAPL'),'headline_text'].iloc[0]
    with pytest.raises(ValueError):fit_frame(frame,'final')
    corrupt=frame.copy();row=corrupt.index[(corrupt.asset_id=='AAPL') & (corrupt.session==pd.Timestamp('2023-12-29'))][0]
    corrupt.at[row,'news_ids']=[news.loc[news.title.eq('new report'),'news_id'].iloc[0]]
    with pytest.raises(AssertionError,match='causal window'):audit(corrupt,news)


def test_missing_prices_are_not_forward_filled():
    prices,dividends,_,schedule=data()
    date=pd.Timestamp('2023-12-28')
    prices=prices[prices.session!=date]
    frame=market_features(prices,dividends,schedule)
    row=frame[(frame.asset_id=='AAPL')&(frame.session==date)].iloc[0]
    assert not row.price_observed and not row.feature_ready and np.isnan(row.close)
    preceding=frame[(frame.asset_id=='AAPL')&(frame.session==pd.Timestamp('2023-12-26'))].iloc[0]
    assert not preceding.target_valid and np.isnan(preceding.target_return)


def test_calendar_nanoseconds_and_date_parser_microseconds_join():
    prices,dividends,treasury,schedule=data()
    schedule['close']=schedule.close.astype('datetime64[ns, UTC]')
    schedule['open']=schedule.open.astype('datetime64[ns, UTC]')
    frame=join_treasury(market_features(prices,dividends,schedule),treasury)
    assert frame.treasury_3m_yield.notna().any()
    assert (frame.treasury_available_at.dropna()<=frame.loc[frame.treasury_available_at.notna(),'decision_at']).all()


def test_dedup_does_not_rewrite_past_news():
    old=articles()
    dup=old.iloc[[0]].copy();dup['source']='z'
    normalized,_=normalize_news(pd.concat([old,dup]))
    assert len(normalized)==len(old)
    later=old.iloc[[0]].copy();later['report_date']='2024-01-04'
    with_later,_=normalize_news(pd.concat([old,later]))
    assert set(normalized.news_id).issubset(set(with_later.news_id))
