"""Causal price/news dataframe construction; labels never enter observations.

Public historical archives are not original point-in-time database vintages.
The builder enforces temporal joins, split purging and feature causality; the
data card separately states the source-vintage limitation.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

UNIVERSE = dict(zip(
    'AAPL AXP BA CAT CSCO CVX DIS DOW GS HD IBM INTC JNJ JPM KO MCD MMM MRK MSFT NKE PFE PG TRV UNH UTX V VZ WBA WMT XOM'.split(),
    ['Apple','American Express','Boeing','Caterpillar','Cisco','Chevron','Walt Disney','Dow',
     'Goldman Sachs','Home Depot','IBM','Intel','Johnson & Johnson','JPMorgan Chase',
     'Coca-Cola',"McDonald’s",'3M','Merck','Microsoft','Nike','Pfizer','Procter & Gamble',
     'Travelers','UnitedHealth','United Technologies','Visa','Verizon',
     'Walgreens Boots Alliance','Walmart','Exxon Mobil']))
FEATURES = ['return_1d','return_5d','return_20d','close_open_return','high_close_ratio',
            'low_close_ratio','volume_change_1d','relative_volume_20d','volatility_20d',
            'treasury_3m_yield','news_count_1d','news_count_7d']
OBSERVATION_COLUMNS = ['asset_id','symbol_asof','session','decision_at','price_observed',
                       'feature_ready','news_ids','headline_text','news_completeness_unknown'] + FEATURES
PERIODS = {'train':('2020-01-01','2024-01-01'), 'valid':('2024-01-01','2025-01-01'),
           'discovery':('2025-01-01','2026-01-01'), 'final':('2026-01-01','2027-01-01')}


def partition(date):
    date = str(date)[:10]
    return next((k for k,(start,end) in PERIODS.items() if start <= date < end), 'context')


def conservative_available_at(value):
    """Use date+2 at 00:00 UTC for all sources, regardless of claimed precision.

    This is after the end of that date even in UTC-12, with a further 12-hour
    buffer. It avoids interpreting missing publication times as midnight.
    It does not establish the original version of subsequently revised text.
    """
    match = re.match(r'^(\d{4}-\d{2}-\d{2})', str(value))
    if not match:
        return pd.NaT
    parsed = pd.to_datetime(match.group(1), utc=True, errors='coerce')
    return parsed + pd.Timedelta(days=2)


def normalize_news(frame):
    frame = frame.copy()
    required = {'symbol','title','report_date','source','link'}
    if not required.issubset(frame.columns):
        raise ValueError(f'missing news columns: {required-set(frame.columns)}')
    frame['asset_id'] = frame.symbol.replace({'RTX':'UTX'})
    frame = frame[frame.asset_id.isin(UNIVERSE)].copy()
    frame['title'] = frame.title.fillna('').astype(str).str.replace(r'\s+', ' ', regex=True).str.strip()
    dates = frame.report_date.astype(str).str.extract(r'^(\d{4}-\d{2}-\d{2})',expand=False)
    frame['available_at'] = (pd.to_datetime(dates,utc=True,errors='coerce')
                             + pd.Timedelta(days=2)).astype('datetime64[ns, UTC]')
    rejected = frame[frame.available_at.isna() | frame.title.eq('')].copy()
    frame = frame[frame.available_at.notna() & frame.title.ne('')].copy()
    frame['news_id'] = [hashlib.sha256((a+'|'+str(t)+'|'+title.casefold()).encode()).hexdigest()
                        for a,t,title in zip(frame.asset_id,frame.available_at,frame.title)]
    # Retain conflicting later-day versions at their own availability time.
    # Never globally remove an older item because a newer duplicate exists.
    frame = frame.sort_values(['available_at','source','news_id']).drop_duplicates('news_id',keep='first')
    return frame.reset_index(drop=True), rejected


def market_features(prices, dividends, schedule):
    """Scale-invariant features of vendor split-adjusted prices and dividends.

    All rolling windows look backward, without filling missing prices. No raw
    vendor price level, future adjustment factor or forward return is an input.
    """
    pieces = []
    for asset in UNIVERSE:
        vendor = 'RTX' if asset == 'UTX' else asset
        p = prices[prices.symbol.eq(vendor)].copy().set_index('session')
        if p.index.duplicated().any():
            raise ValueError(f'duplicate price rows for {asset}')
        p = p.reindex(schedule.index).copy()
        for col in ['open','high','low','close','volume']:
            p[col] = pd.to_numeric(p[col],errors='raise').astype(float)
        observed = p[['open','high','low','close']].notna().all(axis=1)
        bad = observed & ((p[['open','high','low','close']]<=0).any(axis=1)
                          | (p.high < p[['open','close','low']].max(axis=1)-1e-3)
                          | (p.low > p[['open','close','high']].min(axis=1)+1e-3)
                          | (p.volume < 0))
        if bad.any():
            raise ValueError(f'invalid OHLCV for {asset}: {list(p.index[bad][:5])}')
        # Zero-volume synthetic quotes are not evidence of a trading session.
        observed &= p.volume.gt(0)
        p.loc[~observed,['open','high','low','close','volume']] = np.nan
        div = dividends[dividends.symbol.eq(vendor)].groupby('session').amount.sum()
        p['dividend'] = div.reindex(p.index,fill_value=0).astype(float)
        daily = (p.close+p.dividend)/p.close.shift(1)-1
        p['return_1d'] = daily
        for k in [5,20]:
            p[f'return_{k}d'] = (1+daily).rolling(k,min_periods=k).apply(np.prod,raw=True)-1
        p['close_open_return'] = p.close/p.open-1
        p['high_close_ratio'] = p.high/p.close-1
        p['low_close_ratio'] = p.low/p.close-1
        p['volume_change_1d'] = p.volume/p.volume.shift(1)-1
        p['relative_volume_20d'] = p.volume/p.volume.shift(1).rolling(20,min_periods=20).mean()
        p['volatility_20d'] = daily.rolling(20,min_periods=20).std(ddof=1)*np.sqrt(252)
        p['asset_id'] = asset
        p['symbol_asof'] = asset
        if asset == 'UTX':
            p.loc[p.index >= '2020-04-03','symbol_asof'] = 'RTX'
        p['price_observed'] = observed
        p['feature_ready'] = p[FEATURES[:9]].notna().all(axis=1)
        p['decision_at'] = schedule.close + pd.Timedelta(minutes=15)
        p['execution_at'] = schedule.open.shift(-1)
        p['label_known_at'] = schedule.open.shift(-2)
        p['target_return'] = (p.open.shift(-2)+p.dividend.shift(-2))/p.open.shift(-1)-1
        p['target_valid'] = observed.shift(-1,fill_value=False) & observed.shift(-2,fill_value=False)
        p.loc[~p.target_valid,'target_return'] = np.nan
        p['split'] = [partition(d) for d in p.index]
        p['target_split'] = [partition(d) for d in p.label_known_at]
        # Fit labels must mature within their assigned split.
        p['label_released'] = p.target_valid & p.split.eq(p.target_split)
        p.index.name='session'
        pieces.append(p.reset_index())
    return pd.concat(pieces,ignore_index=True).sort_values(['session','asset_id']).reset_index(drop=True)


def join_news(panel, news):
    panel = panel.reset_index(drop=True).copy()
    identifiers = [[] for _ in range(len(panel))]
    texts = ['' for _ in range(len(panel))]
    counts_1d = np.zeros(len(panel),dtype=np.int64)
    counts_7d = np.zeros(len(panel),dtype=np.int64)
    for asset, group in panel.groupby('asset_id',sort=False):
        items = news[news.asset_id.eq(asset)].sort_values(['available_at','news_id'])
        times = items.available_at.astype('datetime64[ns, UTC]').astype('int64').to_numpy()
        ids = items.news_id.tolist()
        titles = items.title.tolist()
        for idx,row in group.iterrows():
            cutoff = row.decision_at.value
            left = np.searchsorted(times,cutoff-pd.Timedelta(days=7).value,side='right')
            right = np.searchsorted(times,cutoff,side='right')
            identifiers[idx] = ids[left:right]
            texts[idx] = '\n'.join(titles[left:right])
            counts_7d[idx] = right-left
            counts_1d[idx] = right-np.searchsorted(times,cutoff-pd.Timedelta(days=1).value,side='right')
    # Construct Arrow-backed string columns once; scalar assignment repeatedly
    # copies the accumulated text column and becomes quadratic on real corpora.
    panel['news_ids'] = identifiers
    panel['headline_text'] = texts
    panel['news_count_1d'] = counts_1d
    panel['news_count_7d'] = counts_7d
    panel['news_completeness_unknown'] = True
    return panel


def join_treasury(panel, treasury):
    series = treasury.copy()
    series['available_at'] = series.report_date.map(conservative_available_at)
    series['available_at'] = series.available_at.astype('datetime64[ns, UTC]')
    series['treasury_3m_yield'] = pd.to_numeric(series.bc_3month,errors='coerce')/100
    series = series.dropna(subset=['available_at','treasury_3m_yield']).sort_values('available_at')
    if series.available_at.duplicated().any():
        raise ValueError('duplicate Treasury dates')
    panel=panel.copy()
    panel['decision_at']=panel.decision_at.astype('datetime64[ns, UTC]')
    joined = pd.merge_asof(panel.sort_values('decision_at'),
        series[['available_at','treasury_3m_yield']].rename(columns={'available_at':'treasury_available_at'}),
        left_on='decision_at',right_on='treasury_available_at',direction='backward',
        tolerance=pd.Timedelta(days=10))
    return joined.sort_values(['session','asset_id']).reset_index(drop=True)


def observation(panel, session):
    """Explicit allowlist: no label, future mask, raw price level or audit field."""
    day = pd.Timestamp(session)
    return panel.loc[panel.session.eq(day),OBSERVATION_COLUMNS].copy()


def fit_frame(panel, split):
    if split not in ('train','valid'):
        raise ValueError('fit may access only inner train/valid')
    rows = panel[panel.split.eq(split)].copy()
    labels = rows.target_return.where(rows.label_released)
    result = rows[OBSERVATION_COLUMNS].copy()
    result['target_return'] = labels
    return result


def audit(panel, news):
    if panel.duplicated(['session','asset_id']).any():
        raise AssertionError('duplicate panel keys')
    if (panel.decision_at >= panel.execution_at).fillna(False).any():
        raise AssertionError('decision does not precede execution')
    if (panel.execution_at >= panel.label_known_at).fillna(False).any():
        raise AssertionError('execution does not precede return realization')
    if (panel.treasury_available_at > panel.decision_at).fillna(False).any():
        raise AssertionError('future Treasury observation')
    lookup = news.set_index('news_id')
    edge_count = 0
    for row in panel.itertuples():
        if not row.news_ids:
            continue
        selected = lookup.loc[row.news_ids]
        if not selected.asset_id.eq(row.asset_id).all():
            raise AssertionError('wrong equity in news join')
        if not ((selected.available_at <= row.decision_at)
                & (selected.available_at > row.decision_at-pd.Timedelta(days=7))).all():
            raise AssertionError('news outside causal window')
        if '\n'.join(selected.title) != row.headline_text:
            raise AssertionError('text and news IDs differ')
        edge_count += len(selected)
    if panel.loc[panel.label_released,'split'].ne(panel.loc[panel.label_released,'target_split']).any():
        raise AssertionError('label crosses split')
    return dict(rows=len(panel),equities=panel.asset_id.nunique(),news=len(news),
                causal_news_edges_checked=edge_count,
                missing_price_rows=int((~panel.price_observed).sum()),
                labels_purged_at_boundaries=int((panel.target_valid & ~panel.label_released).sum()))
